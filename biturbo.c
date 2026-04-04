/*
 * biturbo.c — Zero-dependency BitNet 1.58-bit inference engine
 *             with TurboQuant INT4 KV cache quantization
 *
 * Loads GGUF models directly. Implements the full 15-stage pipeline:
 *   1.  BPE tokenization (from GGUF metadata)
 *   2.  Token embedding (F16)
 *   3.  RMS norm (pre-attention)
 *   4.  BitLinear INT8 input quantization
 *   5.  1.58-bit ternary weights (I2_S group-interleaved)
 *   6.  BitLinear GEMM (INT8 × ternary)
 *   7.  Q/K/V projection (separate matrices)
 *   8.  RoPE positional encoding
 *   9.  KV cache (TurboQuant uniform INT4)
 *   10. Attention score (GQA)
 *   11. Attention sub-norm + output projection
 *   12. FFN: pre-norm + SqReLU gate
 *   13. FFN: sub-norm + down projection
 *   14. Final norm + LM head (tied to embedding)
 *   15. Sampling → decode
 */

#include "biturbo.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ================================================================
 * VOCAB SORT HELPER (for building binary search index)
 * ================================================================ */

static char** bt_sort_vocab_global;
static int bt_sort_vocab_cmp(const void* a, const void* b) {
    return strcmp(bt_sort_vocab_global[*(const int*)a],
                 bt_sort_vocab_global[*(const int*)b]);
}

/* ================================================================
 * MEMORY HELPERS
 * ================================================================ */

static void* bt_calloc(size_t count, size_t size) {
    void* p = calloc(count, size);
    if (!p) {
        fprintf(stderr, "biturbo: OOM allocating %zu bytes\n", count * size);
        exit(1);
    }
    return p;
}

/* ================================================================
 * §1. GGUF PARSER — reads model config, tokenizer, and weight pointers
 *
 * GGUF v3 format:
 *   [magic:u32] [version:u32] [n_tensors:u64] [n_kv:u64]
 *   [metadata kv pairs...]
 *   [tensor info headers...]
 *   [alignment padding]
 *   [tensor data...]
 * ================================================================ */

#define GGUF_MAGIC    0x46554747  /* "GGUF" */
#define GGUF_TYPE_U8  0
#define GGUF_TYPE_I8  1
#define GGUF_TYPE_U16 2
#define GGUF_TYPE_I16 3
#define GGUF_TYPE_U32 4
#define GGUF_TYPE_I32 5
#define GGUF_TYPE_F32 6
#define GGUF_TYPE_BOOL 7
#define GGUF_TYPE_STR 8
#define GGUF_TYPE_ARR 9
#define GGUF_TYPE_U64 10
#define GGUF_TYPE_I64 11
#define GGUF_TYPE_F64 12

/* GGUF tensor types */
#define GGUF_TENSOR_F32  0
#define GGUF_TENSOR_F16  1
#define GGUF_TENSOR_I2S  36

typedef struct {
    const uint8_t* base;
    size_t pos;
    size_t size;
} gguf_reader_t;

static uint32_t rd_u32(gguf_reader_t* r) {
    uint32_t v;
    memcpy(&v, r->base + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t rd_u64(gguf_reader_t* r) {
    uint64_t v;
    memcpy(&v, r->base + r->pos, 8);
    r->pos += 8;
    return v;
}

static float rd_f32(gguf_reader_t* r) {
    float v;
    memcpy(&v, r->base + r->pos, 4);
    r->pos += 4;
    return v;
}

static double rd_f64(gguf_reader_t* r) {
    double v;
    memcpy(&v, r->base + r->pos, 8);
    r->pos += 8;
    return v;
}

/* Read GGUF string: u64 length + bytes (NOT null-terminated) */
typedef struct { const char* str; uint64_t len; } gguf_str_t;

static gguf_str_t rd_str(gguf_reader_t* r) {
    gguf_str_t s;
    s.len = rd_u64(r);
    s.str = (const char*)(r->base + r->pos);
    r->pos += s.len;
    return s;
}

static int str_eq(gguf_str_t s, const char* lit) {
    size_t ll = strlen(lit);
    return s.len == ll && memcmp(s.str, lit, ll) == 0;
}

/* Skip a value of given type */
static void skip_val(gguf_reader_t* r, uint32_t type);

static size_t type_size(uint32_t t) {
    switch (t) {
        case GGUF_TYPE_U8: case GGUF_TYPE_I8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_U16: case GGUF_TYPE_I16: return 2;
        case GGUF_TYPE_U32: case GGUF_TYPE_I32: case GGUF_TYPE_F32: return 4;
        case GGUF_TYPE_U64: case GGUF_TYPE_I64: case GGUF_TYPE_F64: return 8;
        default: return 0;
    }
}

static void skip_val(gguf_reader_t* r, uint32_t type) {
    if (type == GGUF_TYPE_STR) {
        rd_str(r);
    } else if (type == GGUF_TYPE_ARR) {
        uint32_t etype = rd_u32(r);
        uint64_t count = rd_u64(r);
        if (etype == GGUF_TYPE_STR) {
            for (uint64_t i = 0; i < count; i++) rd_str(r);
        } else {
            r->pos += count * type_size(etype);
        }
    } else {
        r->pos += type_size(type);
    }
}

/* Tensor info from GGUF header */
typedef struct {
    const char* name;
    int name_len;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;  /* relative to data section start */
} gguf_tensor_info_t;

/* Find a tensor by name */
static const gguf_tensor_info_t* find_tensor(const gguf_tensor_info_t* tensors,
                                              int n, const char* name) {
    int nl = (int)strlen(name);
    for (int i = 0; i < n; i++) {
        if (tensors[i].name_len == nl &&
            memcmp(tensors[i].name, name, nl) == 0)
            return &tensors[i];
    }
    return NULL;
}

/* ================================================================
 * §2. RMS NORM
 * ================================================================ */

static void rms_norm(float* out, const float* x, const float* weight,
                     int size, float eps) {
    float ss = 0.0f;
#ifdef __ARM_NEON
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vfmaq_f32(acc, v, v);
    }
    ss = vaddvq_f32(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#else
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
#endif
    float scale = 1.0f / sqrtf(ss / (float)size + eps);
    for (int i = 0; i < size; i++) out[i] = x[i] * scale * weight[i];
}

/* ================================================================
 * §3. BITLINEAR INPUT QUANTIZATION (Stage 4)
 *
 * Per-token: scale = 127 / max(|x|), x_q = round(clamp(x * scale))
 * Returns inverse scale for post-GEMM dequantization.
 * ================================================================ */

static float bitlinear_quantize(int8_t* out, const float* x, int size) {
    float amax = 0.0f;
    for (int i = 0; i < size; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax < 1e-10f) amax = 1e-10f;
    float scale = 127.0f / amax;
    for (int i = 0; i < size; i++) {
        int v = (int)roundf(x[i] * scale);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        out[i] = (int8_t)v;
    }
    return amax / 127.0f;  /* inv_scale */
}

/* ================================================================
 * §4. I2_S TERNARY GEMM (Stages 5-6)
 *
 * Group-interleaved 2-bit packing (blocks of 128 elements, 32 bytes):
 *   byte[gp] (gp=0..31):
 *     bits 7-6 → element at gp +  0  (group 0)
 *     bits 5-4 → element at gp + 32  (group 1)
 *     bits 3-2 → element at gp + 64  (group 2)
 *     bits 1-0 → element at gp + 96  (group 3)
 *   2-bit code: 0=-1, 1=0, 2=+1
 *
 * For each output row: sum ternary_weight[col] * int8_input[col]
 * ================================================================ */

static const int8_t I2S_MAP[4] = {-1, 0, 1, 0};

static void i2s_gemv(float* out, const bt_i2s_weight_t* w,
                     const int8_t* x_q, float inv_scale) {
    int rows = w->rows;
    int cols = w->cols;
    float w_scale = w->scale;
    int n_blocks = (cols + 127) / 128;     /* blocks of 128 elements per row */
    int bytes_per_row = n_blocks * 32;     /* 32 bytes per block */

    for (int r = 0; r < rows; r++) {
        const uint8_t* row_data = w->data + (size_t)r * bytes_per_row;
        int32_t acc = 0;
        int col = 0;

        for (int blk = 0; blk < n_blocks; blk++) {
            const uint8_t* bp = row_data + blk * 32;
            int blk_elems = cols - col;
            if (blk_elems > 128) blk_elems = 128;

            int cols0 = blk_elems >= 32  ? 32 : blk_elems;
            int cols1 = blk_elems >= 64  ? 32 : (blk_elems > 32  ? blk_elems - 32  : 0);
            int cols2 = blk_elems >= 96  ? 32 : (blk_elems > 64  ? blk_elems - 64  : 0);
            int cols3 = blk_elems >= 128 ? 32 : (blk_elems > 96  ? blk_elems - 96  : 0);

            for (int gp = 0; gp < 32; gp++) {
                uint8_t b = bp[gp];
                int8_t c0 = I2S_MAP[(b >> 6) & 3];
                int8_t c1 = I2S_MAP[(b >> 4) & 3];
                int8_t c2 = I2S_MAP[(b >> 2) & 3];
                int8_t c3 = I2S_MAP[(b >> 0) & 3];

                if (gp < cols0) acc += (int32_t)c0 * (int32_t)x_q[col + 0*32 + gp];
                if (gp < cols1) acc += (int32_t)c1 * (int32_t)x_q[col + 1*32 + gp];
                if (gp < cols2) acc += (int32_t)c2 * (int32_t)x_q[col + 2*32 + gp];
                if (gp < cols3) acc += (int32_t)c3 * (int32_t)x_q[col + 3*32 + gp];
            }
            col += blk_elems;
        }
        out[r] = (float)acc * inv_scale * w_scale;
    }
}

/* BitLinear forward: quantize input → I2_S GEMV → output */
static void bitlinear_forward(float* out, const float* x, int8_t* q8_buf,
                              const bt_i2s_weight_t* w) {
    float inv_scale = bitlinear_quantize(q8_buf, x, w->cols);
    i2s_gemv(out, w, q8_buf, inv_scale);
}

/* ================================================================
 * §5. ROPE (Stage 8)
 * ================================================================ */

static void rope(float* vec, int head_dim, int n_heads, int pos, float theta) {
    for (int h = 0; h < n_heads; h++) {
        float* hv = vec + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float v0 = hv[i], v1 = hv[i + 1];
            hv[i]     = v0 * cos_a - v1 * sin_a;
            hv[i + 1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

/* ================================================================
 * §6. KV CACHE — TurboQuant INT4 (Stage 9)
 * ================================================================ */

static void kv_quantize_int4(bt_kv_block_t* blocks, const float* src,
                             int dim, int blocks_per_head) {
    for (int b = 0; b < blocks_per_head; b++) {
        bt_kv_block_t* blk = &blocks[b];
        int offset = b * BT_QK;
        int count = dim - offset;
        if (count > BT_QK) count = BT_QK;

        float mn = src[offset], mx = src[offset];
        for (int i = 1; i < count; i++) {
            float v = src[offset + i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        float range = mx - mn;
        if (range < 1e-8f) range = 1e-8f;
        float scale = range / 16.0f;

        blk->scale = bt_f32_to_f16(scale);
        blk->zero_point = bt_f32_to_f16(mn);

        memset(blk->qs, 0, BT_QK / 2);
        for (int i = 0; i < count; i++) {
            int q = (int)floorf((src[offset + i] - mn) / scale);
            if (q < 0) q = 0;
            if (q > 15) q = 15;
            if (i % 2 == 0)
                blk->qs[i / 2] = (uint8_t)q;
            else
                blk->qs[i / 2] |= (uint8_t)(q << 4);
        }
    }
}

/* Dot product: float query with INT4-quantized key */
static float dot_q_kv4(const float* query, const bt_kv_block_t* blocks,
                       int head_dim, int bph) {
    float dot = 0.0f;
    for (int b = 0; b < bph; b++) {
        const bt_kv_block_t* blk = &blocks[b];
        float sc = bt_f16_to_f32(blk->scale);
        float mn = bt_f16_to_f32(blk->zero_point);
        int off = b * BT_QK;
        int cnt = head_dim - off;
        if (cnt > BT_QK) cnt = BT_QK;
        for (int i = 0; i < cnt; i++) {
            uint8_t byte = blk->qs[i / 2];
            int q = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            dot += query[off + i] * (mn + ((float)q + 0.5f) * sc);
        }
    }
    return dot;
}

/* Weighted accumulate: out += weight * dequantized INT4 vector */
static void accum_kv4(float* out, float weight, const bt_kv_block_t* blocks,
                      int head_dim, int bph) {
    for (int b = 0; b < bph; b++) {
        const bt_kv_block_t* blk = &blocks[b];
        float sc = bt_f16_to_f32(blk->scale);
        float mn = bt_f16_to_f32(blk->zero_point);
        int off = b * BT_QK;
        int cnt = head_dim - off;
        if (cnt > BT_QK) cnt = BT_QK;
        for (int i = 0; i < cnt; i++) {
            uint8_t byte = blk->qs[i / 2];
            int q = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            out[off + i] += weight * (mn + ((float)q + 0.5f) * sc);
        }
    }
}

/* ================================================================
 * §7. SOFTMAX
 * ================================================================ */

static void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < size; i++) x[i] *= inv;
}

/* ================================================================
 * §8. FORWARD PASS (Stages 2-14)
 * ================================================================ */

void bt_forward(bt_model_t* model, int token, int pos) {
    bt_config_t* cfg = &model->config;
    bt_weights_t* w = &model->weights;
    bt_state_t* s = &model->state;

    int dim = cfg->dim;
    int head_dim = BT_HEAD_DIM(cfg);
    int n_heads = cfg->n_heads;
    int n_kv_heads = cfg->n_kv_heads;
    int gqa_groups = BT_GQA_GROUPS(cfg);
    int ffn_dim = cfg->ffn_dim;

    /* ── Stage 2: Token embedding (F16 → F32) ── */
    const uint16_t* emb = w->token_embedding + (size_t)token * dim;
    for (int i = 0; i < dim; i++)
        s->x[i] = bt_f16_to_f32(emb[i]);

    /* ── Transformer layers ── */
    for (int l = 0; l < cfg->n_layers; l++) {
        bt_layer_weights_t* lw = &w->layers[l];
        bt_kv_cache_t* kv = &s->kv[l];
        int bph = kv->blocks_per_head;

        /* Stage 3: Pre-attention RMS norm */
        rms_norm(s->xb, s->x, lw->attn_norm, dim, cfg->norm_eps);

        /* Stage 4-6-7: Q/K/V projections (BitLinear) */
        float inv_scale = bitlinear_quantize(s->q8_buf, s->xb, dim);
        i2s_gemv(s->q, &lw->wq, s->q8_buf, inv_scale);
        i2s_gemv(s->k, &lw->wk, s->q8_buf, inv_scale);
        i2s_gemv(s->v, &lw->wv, s->q8_buf, inv_scale);

        /* Stage 8: RoPE on Q and K */
        rope(s->q, head_dim, n_heads, pos, cfg->rope_theta);
        rope(s->k, head_dim, n_kv_heads, pos, cfg->rope_theta);

        /* Stage 9: Store K/V into INT4 quantized cache */
        for (int h = 0; h < n_kv_heads; h++) {
            size_t idx = (size_t)h * cfg->max_seq_len * bph + (size_t)pos * bph;
            kv_quantize_int4(&kv->k_cache[idx], s->k + h * head_dim,
                             head_dim, bph);
            kv_quantize_int4(&kv->v_cache[idx], s->v + h * head_dim,
                             head_dim, bph);
        }

        /* Stage 10: Attention scores (GQA) */
        int seq_len = pos + 1;
        float att_scale = 1.0f / sqrtf((float)head_dim);
        for (int qh = 0; qh < n_heads; qh++) {
            float* qhead = s->q + qh * head_dim;
            int kv_head = qh / gqa_groups;
            float* att = s->att + (size_t)qh * cfg->max_seq_len;

            for (int t = 0; t < seq_len; t++) {
                size_t ki = (size_t)kv_head * cfg->max_seq_len * bph
                          + (size_t)t * bph;
                att[t] = dot_q_kv4(qhead, &kv->k_cache[ki], head_dim, bph)
                       * att_scale;
            }
            softmax(att, seq_len);

            /* Weighted V sum → xb2[qh * head_dim] */
            float* out_h = s->xb2 + qh * head_dim;
            memset(out_h, 0, head_dim * sizeof(float));
            for (int t = 0; t < seq_len; t++) {
                if (att[t] < 1e-8f) continue;
                size_t vi = (size_t)kv_head * cfg->max_seq_len * bph
                          + (size_t)t * bph;
                accum_kv4(out_h, att[t], &kv->v_cache[vi], head_dim, bph);
            }
        }

        /* Stage 11: Attention sub-norm + output projection */
        rms_norm(s->xb, s->xb2, lw->attn_sub_norm, dim, cfg->norm_eps);
        bitlinear_forward(s->xb2, s->xb, s->q8_buf, &lw->wo);

        /* Residual */
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        /* Stage 12: FFN pre-norm + gate/up projections */
        rms_norm(s->xb, s->x, lw->ffn_norm, dim, cfg->norm_eps);
        inv_scale = bitlinear_quantize(s->q8_buf, s->xb, dim);
        i2s_gemv(s->hb, &lw->w_gate, s->q8_buf, inv_scale);
        i2s_gemv(s->hb2, &lw->w_up, s->q8_buf, inv_scale);

        /* SqReLU gating: hidden = SqReLU(gate) * up */
        for (int i = 0; i < ffn_dim; i++) {
            float g = s->hb[i];
            float r = (g > 0.0f) ? g : 0.0f;
            s->hb[i] = r * r * s->hb2[i];
        }

        /* Stage 13: FFN sub-norm + down projection */
        rms_norm(s->hb2, s->hb, lw->ffn_sub_norm, ffn_dim, cfg->norm_eps);
        bitlinear_forward(s->xb, s->hb2, s->q8_buf, &lw->w_down);

        /* Residual */
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    /* Stage 14: Final norm + LM head (tied to token embedding) */
    rms_norm(s->xb, s->x, w->final_norm, dim, cfg->norm_eps);

    /* LM head: logits[v] = dot(xb, token_embd[v]) for all vocab
     * token_embd is F16 [vocab_size][dim] */
    for (int v = 0; v < cfg->vocab_size; v++) {
        const uint16_t* ev = w->token_embedding + (size_t)v * dim;
        float sum = 0.0f;
        for (int d = 0; d < dim; d++)
            sum += s->xb[d] * bt_f16_to_f32(ev[d]);
        s->logits[v] = sum;
    }
}

/* ================================================================
 * §9. SAMPLING (Stage 15)
 * ================================================================ */

void bt_sampler_init(bt_sampler_t* s, float temperature, float top_p,
                     uint64_t seed) {
    s->temperature = temperature;
    s->top_p = top_p;
    s->rng_state = seed ? seed : (uint64_t)time(NULL);
}

static float bt_random(bt_sampler_t* s) {
    uint64_t x = s->rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    s->rng_state = x;
    return (float)((x * 0x2545F4914F6CDD1DULL) >> 40) / (float)(1 << 24);
}

typedef struct { float prob; int idx; } bt_pi_t;
static int pi_cmp(const void* a, const void* b) {
    float pa = ((const bt_pi_t*)a)->prob;
    float pb = ((const bt_pi_t*)b)->prob;
    return (pa < pb) - (pa > pb);
}

int bt_sample(bt_sampler_t* s, const float* logits, int vocab_size) {
    if (s->temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    float* probs = (float*)bt_calloc(vocab_size, sizeof(float));
    for (int i = 0; i < vocab_size; i++)
        probs[i] = logits[i] / s->temperature;
    softmax(probs, vocab_size);

    int result = 0;
    if (s->top_p < 1.0f) {
        bt_pi_t* sorted = (bt_pi_t*)bt_calloc(vocab_size, sizeof(bt_pi_t));
        for (int i = 0; i < vocab_size; i++) {
            sorted[i].prob = probs[i];
            sorted[i].idx = i;
        }
        qsort(sorted, vocab_size, sizeof(bt_pi_t), pi_cmp);

        float cumsum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += sorted[i].prob;
            if (cumsum >= s->top_p) { cutoff = i + 1; break; }
        }
        float r = bt_random(s) * cumsum;
        float cdf = 0.0f;
        result = sorted[0].idx;
        for (int i = 0; i < cutoff; i++) {
            cdf += sorted[i].prob;
            if (r <= cdf) { result = sorted[i].idx; break; }
        }
        free(sorted);
    } else {
        float r = bt_random(s);
        float cdf = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cdf += probs[i];
            if (r <= cdf) { result = i; break; }
        }
    }
    free(probs);
    return result;
}

/* ================================================================
 * §10. TOKENIZER (Stage 1) — BPE from GGUF metadata
 *
 * GPT-2 BPE uses a byte-to-unicode mapping so all 256 byte values
 * can be represented as unique Unicode characters in the vocabulary:
 *   - Printable ASCII 0x21-0x7E → identity (same codepoint)
 *   - Latin-1 0xA1-0xAC, 0xAE-0xFF → identity
 *   - Everything else (0x00-0x20, 0x7F-0xA0, 0xAD) → U+0100..U+0143
 * Notably: space (0x20) → U+0120 (Ġ), newline (0x0A) → U+010A (Ċ)
 *
 * Encoding: raw bytes → byte-to-unicode → per-byte vocab lookup → BPE merge
 * Decoding: token string → unicode-to-byte reverse mapping → raw bytes
 * ================================================================ */

/* Convert a raw byte to its GPT-2 BPE UTF-8 string (1-3 bytes + NUL).
 * Returns the number of bytes written (excluding NUL). */
static int byte_to_bpe(unsigned char byte, char out[4]) {
    if (byte >= 0x21 && byte <= 0x7E) {
        /* Printable ASCII: maps to itself */
        out[0] = (char)byte;
        out[1] = '\0';
        return 1;
    }
    /* Compute the Unicode codepoint for this byte */
    int cp;
    if ((byte >= 0xA1 && byte <= 0xAC) || (byte >= 0xAE)) {
        /* Latin-1 printable: identity mapping */
        cp = byte;
    } else if (byte <= 0x20) {
        /* 0x00-0x20 (includes space) → U+0100..U+0120 */
        cp = 0x100 + byte;
    } else if (byte == 0x7F) {
        cp = 0x121;  /* DEL → U+0121 */
    } else if (byte >= 0x80 && byte <= 0xA0) {
        cp = 0x122 + (byte - 0x80);  /* 0x80-0xA0 → U+0122..U+0142 */
    } else {
        cp = 0x143;  /* 0xAD (soft hyphen) → U+0143 */
    }
    /* Encode codepoint as UTF-8 (all are <= U+0143, so 2 bytes) */
    out[0] = (char)(0xC0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3F));
    out[2] = '\0';
    return 2;
}

/* Reverse mapping: BPE codepoints U+0100..U+0143 → raw byte values */
static const uint8_t BPE_N2B[68] = {
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07, /* U+0100-0107 */
    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F, /* U+0108-010F */
    0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17, /* U+0110-0117 */
    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F, /* U+0118-011F */
    0x20,                                      /* U+0120 = space */
    0x7F,                                      /* U+0121 = DEL */
    0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87, /* U+0122-0129 */
    0x88,0x89,0x8A,0x8B,0x8C,0x8D,0x8E,0x8F, /* U+012A-0131 */
    0x90,0x91,0x92,0x93,0x94,0x95,0x96,0x97, /* U+0132-0139 */
    0x98,0x99,0x9A,0x9B,0x9C,0x9D,0x9E,0x9F, /* U+013A-0141 */
    0xA0,                                      /* U+0142 = NBSP */
    0xAD,                                      /* U+0143 = soft hyphen */
};

/* Binary search vocab lookup using sorted index. O(log V). */
static int tok_lookup(const bt_tokenizer_t* tok, const char* str) {
    if (!tok->sorted_idx) return -1;
    int lo = 0, hi = tok->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = strcmp(str, tok->vocab[tok->sorted_idx[mid]]);
        if (cmp == 0) return tok->sorted_idx[mid];
        if (cmp < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return -1;
}

int bt_encode(const bt_tokenizer_t* tok, const char* text,
              int* tokens, int max_tokens, int add_bos) {
    if (!text || !tokens) return 0;

    int text_len = (int)strlen(text);
    if (text_len == 0 && !add_bos) return 0;

    /* Work buffer for per-byte tokens before BPE merging */
    int* work = (int*)malloc((text_len + 1) * sizeof(int));
    if (!work) return 0;
    int n_work = 0;

    /* Step 1: Convert each raw byte to its BPE unicode token.
     * "Where is" → bytes [57 68 65 72 65 20 69 73]
     *            → BPE   [W  h  e  r  e  Ġ  i  s ]
     *            → tokens [54 71 68 81 68 220 72 82] */
    for (int i = 0; i < text_len; i++) {
        char bpe[4];
        byte_to_bpe((unsigned char)text[i], bpe);
        int id = tok_lookup(tok, bpe);
        if (id >= 0) {
            work[n_work++] = id;
        }
        /* If not found, skip byte (shouldn't happen with valid vocab) */
    }

    /* Step 2: BPE merge loop — greedily merge the highest-scored adjacent pair.
     * [W, h, e, r, e, Ġ, i, s] → [Wh, e, r, e, Ġ, is]
     * → [Wh, ere, Ġ, is] → [Where, Ġis] → done (no more merges) */
    char merge_buf[1024];
    while (n_work >= 2) {
        float best_score = -1e30f;
        int best_idx = -1, best_tok = -1;

        for (int i = 0; i < n_work - 1; i++) {
            const char* s1 = tok->vocab[work[i]];
            const char* s2 = tok->vocab[work[i + 1]];
            int l1 = (int)strlen(s1);
            int l2 = (int)strlen(s2);
            if (l1 + l2 >= (int)sizeof(merge_buf)) continue;
            memcpy(merge_buf, s1, l1);
            memcpy(merge_buf + l1, s2, l2);
            merge_buf[l1 + l2] = '\0';

            int id = tok_lookup(tok, merge_buf);
            if (id >= 0 && tok->scores[id] > best_score) {
                best_score = tok->scores[id];
                best_idx = i;
                best_tok = id;
            }
        }
        if (best_idx < 0) break;

        work[best_idx] = best_tok;
        for (int i = best_idx + 1; i < n_work - 1; i++)
            work[i] = work[i + 1];
        n_work--;
    }

    /* Step 3: Copy to output with optional BOS prefix */
    int n = 0;
    if (add_bos && n < max_tokens)
        tokens[n++] = tok->bos_id;
    for (int i = 0; i < n_work && n < max_tokens; i++)
        tokens[n++] = work[i];

    free(work);
    return n;
}

/* Decode token ID to raw text.
 * Reverses the GPT-2 byte-to-unicode mapping:
 *   Ġ (U+0120) → space, Ċ (U+010A) → newline, etc.
 * All codepoints in U+0100..U+0143 are looked up in BPE_N2B. */
const char* bt_decode(const bt_tokenizer_t* tok, int prev_token, int token) {
    static char buf[1024];
    if (token < 0 || token >= tok->vocab_size) return "";
    const unsigned char* src = (const unsigned char*)tok->vocab[token];

    int j = 0;
    while (*src && j < (int)sizeof(buf) - 4) {
        if (src[0] < 0x80) {
            /* ASCII: identity mapping */
            buf[j++] = (char)src[0];
            src++;
        } else if ((src[0] & 0xE0) == 0xC0 && (src[1] & 0xC0) == 0x80) {
            /* 2-byte UTF-8: decode codepoint */
            int cp = ((src[0] & 0x1F) << 6) | (src[1] & 0x3F);
            if (cp >= 0x100 && cp <= 0x143) {
                /* BPE remapped byte: look up raw value */
                buf[j++] = (char)BPE_N2B[cp - 0x100];
            } else if ((cp >= 0xA1 && cp <= 0xAC) || (cp >= 0xAE && cp <= 0xFF)) {
                /* Latin-1 identity: output as raw byte */
                buf[j++] = (char)cp;
            } else {
                /* Other 2-byte codepoint: pass through as UTF-8 */
                buf[j++] = (char)src[0];
                buf[j++] = (char)src[1];
            }
            src += 2;
        } else {
            /* 3+ byte UTF-8: pass through */
            buf[j++] = (char)*src++;
        }
    }
    buf[j] = '\0';

    const char* out = buf;
    if (prev_token == tok->bos_id && out[0] == ' ') out++;
    return out;
}

/* ================================================================
 * §11. GGUF MODEL LOADING
 * ================================================================ */

static void init_i2s_weight(bt_i2s_weight_t* w, const gguf_tensor_info_t* ti,
                            const uint8_t* data_base) {
    w->cols = (int)ti->dims[0];  /* ne0 = input features */
    w->rows = (int)ti->dims[1];  /* ne1 = output features */

    const uint8_t* blob = data_base + ti->offset;
    size_t n_elem = (size_t)w->rows * w->cols;
    size_t packed_bytes = n_elem / 4;  /* 4 values per byte */

    w->data = blob;
    /* Scale is the last 4 bytes of the I2_S blob */
    memcpy(&w->scale, blob + packed_bytes, sizeof(float));
}

static float* load_f32_tensor(const gguf_tensor_info_t* ti,
                               const uint8_t* data_base) {
    size_t n = 1;
    for (uint32_t d = 0; d < ti->n_dims; d++) n *= ti->dims[d];
    float* out = (float*)bt_calloc(n, sizeof(float));
    memcpy(out, data_base + ti->offset, n * sizeof(float));
    return out;
}

static int alloc_state(bt_state_t* s, const bt_config_t* cfg) {
    int dim = cfg->dim;
    int kv_dim = BT_KV_DIM(cfg);
    int ffn_dim = cfg->ffn_dim;
    int max_buf = dim;
    if (kv_dim > max_buf) max_buf = kv_dim;
    if (ffn_dim > max_buf) max_buf = ffn_dim;

    s->x      = (float*)bt_calloc(dim, sizeof(float));
    s->xb     = (float*)bt_calloc(dim, sizeof(float));
    s->xb2    = (float*)bt_calloc(dim, sizeof(float));
    s->hb     = (float*)bt_calloc(ffn_dim, sizeof(float));
    s->hb2    = (float*)bt_calloc(ffn_dim, sizeof(float));
    s->q      = (float*)bt_calloc(dim, sizeof(float));
    s->k      = (float*)bt_calloc(kv_dim, sizeof(float));
    s->v      = (float*)bt_calloc(kv_dim, sizeof(float));
    s->att    = (float*)bt_calloc((size_t)cfg->n_heads * cfg->max_seq_len,
                                  sizeof(float));
    s->q8_buf = (int8_t*)bt_calloc(max_buf, sizeof(int8_t));
    s->logits = (float*)bt_calloc(cfg->vocab_size, sizeof(float));

    int head_dim = BT_HEAD_DIM(cfg);
    int bph = (head_dim + BT_QK - 1) / BT_QK;

    s->kv = (bt_kv_cache_t*)bt_calloc(cfg->n_layers, sizeof(bt_kv_cache_t));
    for (int l = 0; l < cfg->n_layers; l++) {
        s->kv[l].blocks_per_head = bph;
        size_t total = (size_t)cfg->n_kv_heads * cfg->max_seq_len * bph;
        s->kv[l].k_cache = (bt_kv_block_t*)bt_calloc(total, sizeof(bt_kv_block_t));
        s->kv[l].v_cache = (bt_kv_block_t*)bt_calloc(total, sizeof(bt_kv_block_t));
    }
    return 0;
}

int bt_load_model(bt_model_t* model, const char* path) {
    memset(model, 0, sizeof(*model));

    /* mmap the GGUF file */
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "biturbo: cannot open '%s'\n", path);
        return -1;
    }
    struct stat st;
    fstat(fd, &st);
    model->mmap_size = (size_t)st.st_size;
    model->mmap_data = (uint8_t*)mmap(NULL, model->mmap_size,
                                       PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (model->mmap_data == MAP_FAILED) {
        fprintf(stderr, "biturbo: mmap failed\n");
        return -1;
    }

    gguf_reader_t r = { model->mmap_data, 0, model->mmap_size };

    /* Header */
    uint32_t magic = rd_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "biturbo: not a GGUF file\n");
        munmap(model->mmap_data, model->mmap_size);
        return -1;
    }
    uint32_t version = rd_u32(&r);
    if (version < 2 || version > 3) {
        fprintf(stderr, "biturbo: unsupported GGUF version %u\n", version);
        munmap(model->mmap_data, model->mmap_size);
        return -1;
    }
    uint64_t n_tensors = rd_u64(&r);
    uint64_t n_kv = rd_u64(&r);

    fprintf(stderr, "biturbo: GGUF v%u — %llu tensors, %llu metadata\n",
            version, (unsigned long long)n_tensors, (unsigned long long)n_kv);

    /* Parse metadata → extract config and tokenizer */
    bt_config_t* cfg = &model->config;
    cfg->max_seq_len = 4096;  /* default */

    bt_tokenizer_t* tok = &model->tokenizer;
    tok->bos_id = -1;
    tok->eos_id = -1;
    tok->eot_id = -1;

    /* We need to do two passes or save positions for arrays.
     * For token arrays, save reader position and parse later. */
    size_t tok_tokens_pos = 0;
    uint64_t tok_tokens_count = 0;
    size_t tok_scores_pos = 0;
    uint64_t tok_scores_count = 0;
    (void)0; /* tok_types: reserved for future use */

    for (uint64_t i = 0; i < n_kv; i++) {
        gguf_str_t key = rd_str(&r);
        uint32_t vtype = rd_u32(&r);

        /* Model config */
        if (str_eq(key, "bitnet-b1.58.embedding_length") && vtype == GGUF_TYPE_U32) {
            cfg->dim = (int)rd_u32(&r);
        } else if (str_eq(key, "bitnet-b1.58.block_count") && vtype == GGUF_TYPE_U32) {
            cfg->n_layers = (int)rd_u32(&r);
        } else if (str_eq(key, "bitnet-b1.58.attention.head_count") && vtype == GGUF_TYPE_U32) {
            cfg->n_heads = (int)rd_u32(&r);
        } else if (str_eq(key, "bitnet-b1.58.attention.head_count_kv") && vtype == GGUF_TYPE_U32) {
            cfg->n_kv_heads = (int)rd_u32(&r);
        } else if (str_eq(key, "bitnet-b1.58.vocab_size") && vtype == GGUF_TYPE_U32) {
            cfg->vocab_size = (int)rd_u32(&r);
        } else if (str_eq(key, "bitnet-b1.58.feed_forward_length") && vtype == GGUF_TYPE_U32) {
            cfg->ffn_dim = (int)rd_u32(&r);
        } else if (str_eq(key, "bitnet-b1.58.context_length") && vtype == GGUF_TYPE_U32) {
            cfg->max_seq_len = (int)rd_u32(&r);
        } else if (str_eq(key, "bitnet-b1.58.attention.layer_norm_rms_epsilon") && vtype == GGUF_TYPE_F32) {
            cfg->norm_eps = rd_f32(&r);
        } else if (str_eq(key, "bitnet-b1.58.rope.freq_base") && vtype == GGUF_TYPE_F32) {
            cfg->rope_theta = rd_f32(&r);
        } else if (str_eq(key, "bitnet-b1.58.rope.freq_base") && vtype == GGUF_TYPE_F64) {
            cfg->rope_theta = (float)rd_f64(&r);
        /* Tokenizer arrays: save positions for later */
        } else if (str_eq(key, "tokenizer.ggml.tokens") && vtype == GGUF_TYPE_ARR) {
            uint32_t etype = rd_u32(&r);
            tok_tokens_count = rd_u64(&r);
            tok_tokens_pos = r.pos;
            /* Skip the string array */
            if (etype == GGUF_TYPE_STR) {
                for (uint64_t j = 0; j < tok_tokens_count; j++) rd_str(&r);
            }
        } else if (str_eq(key, "tokenizer.ggml.scores") && vtype == GGUF_TYPE_ARR) {
            uint32_t etype = rd_u32(&r);
            tok_scores_count = rd_u64(&r);
            tok_scores_pos = r.pos;
            r.pos += tok_scores_count * type_size(etype);
        } else if (str_eq(key, "tokenizer.ggml.bos_token_id") && vtype == GGUF_TYPE_U32) {
            tok->bos_id = (int)rd_u32(&r);
        } else if (str_eq(key, "tokenizer.ggml.eos_token_id") && vtype == GGUF_TYPE_U32) {
            tok->eos_id = (int)rd_u32(&r);
        } else {
            skip_val(&r, vtype);
        }
    }

    fprintf(stderr, "biturbo: dim=%d layers=%d heads=%d kv_heads=%d "
            "vocab=%d ffn=%d ctx=%d\n",
            cfg->dim, cfg->n_layers, cfg->n_heads, cfg->n_kv_heads,
            cfg->vocab_size, cfg->ffn_dim, cfg->max_seq_len);

    /* Build tokenizer from saved positions */
    if (tok_tokens_count > 0) {
        tok->vocab_size = (int)tok_tokens_count;
        tok->vocab = (char**)bt_calloc(tok->vocab_size, sizeof(char*));
        tok->scores = (float*)bt_calloc(tok->vocab_size, sizeof(float));
        tok->max_token_len = 0;

        /* Read token strings */
        gguf_reader_t tr = { model->mmap_data, tok_tokens_pos, model->mmap_size };
        for (int i = 0; i < tok->vocab_size; i++) {
            gguf_str_t s = rd_str(&tr);
            tok->vocab[i] = (char*)bt_calloc(s.len + 1, 1);
            memcpy(tok->vocab[i], s.str, s.len);
            tok->vocab[i][s.len] = '\0';
            if ((int)s.len > tok->max_token_len)
                tok->max_token_len = (int)s.len;
        }

        /* Read scores */
        if (tok_scores_count > 0) {
            gguf_reader_t sr = { model->mmap_data, tok_scores_pos, model->mmap_size };
            for (int i = 0; i < tok->vocab_size && i < (int)tok_scores_count; i++) {
                tok->scores[i] = rd_f32(&sr);
            }
        }

        /* Find special tokens by content if not in metadata */
        if (tok->bos_id < 0) {
            for (int i = 0; i < tok->vocab_size; i++) {
                if (strcmp(tok->vocab[i], "<|begin_of_text|>") == 0 ||
                    strcmp(tok->vocab[i], "<s>") == 0) {
                    tok->bos_id = i; break;
                }
            }
        }
        if (tok->eos_id < 0) {
            for (int i = 0; i < tok->vocab_size; i++) {
                if (strcmp(tok->vocab[i], "<|end_of_text|>") == 0 ||
                    strcmp(tok->vocab[i], "</s>") == 0) {
                    tok->eos_id = i; break;
                }
            }
        }
        for (int i = 0; i < tok->vocab_size; i++) {
            if (strcmp(tok->vocab[i], "<|eot_id|>") == 0) {
                tok->eot_id = i; break;
            }
        }

        /* Build sorted vocab index for O(log V) binary search */
        tok->sorted_idx = (int*)bt_calloc(tok->vocab_size, sizeof(int));
        for (int i = 0; i < tok->vocab_size; i++) tok->sorted_idx[i] = i;
        bt_sort_vocab_global = tok->vocab;
        qsort(tok->sorted_idx, tok->vocab_size, sizeof(int), bt_sort_vocab_cmp);

        fprintf(stderr, "biturbo: tokenizer: %d tokens, max_len=%d, "
                "bos=%d eos=%d\n",
                tok->vocab_size, tok->max_token_len, tok->bos_id, tok->eos_id);
    }

    /* Read tensor info headers */
    gguf_tensor_info_t* tensors = NULL;
    tensors = (gguf_tensor_info_t*)bt_calloc(
        n_tensors, sizeof(gguf_tensor_info_t));
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_str_t name = rd_str(&r);
        tensors[i].name = name.str;
        tensors[i].name_len = (int)name.len;
        tensors[i].n_dims = rd_u32(&r);
        for (uint32_t d = 0; d < tensors[i].n_dims; d++)
            tensors[i].dims[d] = rd_u64(&r);
        tensors[i].type = rd_u32(&r);
        tensors[i].offset = rd_u64(&r);
    }

    /* Data section starts at aligned offset */
    size_t alignment = 32;
    size_t data_offset = r.pos;
    data_offset += (alignment - data_offset % alignment) % alignment;
    const uint8_t* data_base = model->mmap_data + data_offset;

    /* Allocate state */
    alloc_state(&model->state, cfg);

    /* Load weights */
    bt_weights_t* wt = &model->weights;
    int n_t = (int)n_tensors;

    /* Token embedding (F16, mmap'd directly) */
    const gguf_tensor_info_t* te = find_tensor(tensors, n_t, "token_embd.weight");
    if (!te) { fprintf(stderr, "biturbo: missing token_embd.weight\n"); goto fail; }
    wt->token_embedding = (const uint16_t*)(data_base + te->offset);

    /* Final norm */
    const gguf_tensor_info_t* fn = find_tensor(tensors, n_t, "output_norm.weight");
    if (!fn) { fprintf(stderr, "biturbo: missing output_norm.weight\n"); goto fail; }
    wt->final_norm = load_f32_tensor(fn, data_base);

    /* Per-layer weights */
    wt->layers = (bt_layer_weights_t*)bt_calloc(cfg->n_layers,
                                                 sizeof(bt_layer_weights_t));
    for (int l = 0; l < cfg->n_layers; l++) {
        bt_layer_weights_t* lw = &wt->layers[l];
        char nm[128];
        const gguf_tensor_info_t* ti;

        /* Helper: find tensor or fail */
        #define MUST_FIND(pattern) do { \
            snprintf(nm, sizeof(nm), pattern, l); \
            ti = find_tensor(tensors, n_t, nm); \
            if (!ti) { fprintf(stderr, "biturbo: missing %s\n", nm); goto fail; } \
        } while(0)

        MUST_FIND("blk.%d.attn_norm.weight");
        lw->attn_norm = load_f32_tensor(ti, data_base);

        MUST_FIND("blk.%d.attn_q.weight");
        init_i2s_weight(&lw->wq, ti, data_base);
        MUST_FIND("blk.%d.attn_k.weight");
        init_i2s_weight(&lw->wk, ti, data_base);
        MUST_FIND("blk.%d.attn_v.weight");
        init_i2s_weight(&lw->wv, ti, data_base);

        MUST_FIND("blk.%d.attn_sub_norm.weight");
        lw->attn_sub_norm = load_f32_tensor(ti, data_base);

        MUST_FIND("blk.%d.attn_output.weight");
        init_i2s_weight(&lw->wo, ti, data_base);

        MUST_FIND("blk.%d.ffn_norm.weight");
        lw->ffn_norm = load_f32_tensor(ti, data_base);

        MUST_FIND("blk.%d.ffn_gate.weight");
        init_i2s_weight(&lw->w_gate, ti, data_base);
        MUST_FIND("blk.%d.ffn_up.weight");
        init_i2s_weight(&lw->w_up, ti, data_base);

        MUST_FIND("blk.%d.ffn_sub_norm.weight");
        lw->ffn_sub_norm = load_f32_tensor(ti, data_base);

        MUST_FIND("blk.%d.ffn_down.weight");
        init_i2s_weight(&lw->w_down, ti, data_base);

        #undef MUST_FIND
    }

    free(tensors);
    fprintf(stderr, "biturbo: model loaded (%.1f MB mmap'd)\n",
            (double)model->mmap_size / (1024 * 1024));
    return 0;

fail:
    free(tensors);
    bt_free_model(model);
    return -1;
}

void bt_free_model(bt_model_t* model) {
    /* Free allocated weight copies (F32 norms) */
    bt_weights_t* wt = &model->weights;
    if (wt->layers) {
        for (int l = 0; l < model->config.n_layers; l++) {
            free(wt->layers[l].attn_norm);
            free(wt->layers[l].attn_sub_norm);
            free(wt->layers[l].ffn_norm);
            free(wt->layers[l].ffn_sub_norm);
        }
        free(wt->layers);
    }
    free(wt->final_norm);

    /* Free state */
    bt_state_t* s = &model->state;
    free(s->x); free(s->xb); free(s->xb2);
    free(s->hb); free(s->hb2);
    free(s->q); free(s->k); free(s->v);
    free(s->att); free(s->q8_buf); free(s->logits);
    if (s->kv) {
        for (int l = 0; l < model->config.n_layers; l++) {
            free(s->kv[l].k_cache);
            free(s->kv[l].v_cache);
        }
        free(s->kv);
    }

    /* Free tokenizer */
    bt_tokenizer_t* tok = &model->tokenizer;
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->vocab[i]);
        free(tok->vocab);
    }
    free(tok->scores);
    free(tok->sorted_idx);

    /* Unmap */
    if (model->mmap_data && model->mmap_data != MAP_FAILED)
        munmap(model->mmap_data, model->mmap_size);

    memset(model, 0, sizeof(*model));
}

/* ================================================================
 * §12. TEXT GENERATION
 * ================================================================ */

void bt_generate(bt_model_t* model, bt_sampler_t* sampler,
                 const char* prompt, int max_tokens) {
    bt_tokenizer_t* tok = &model->tokenizer;

    int* prompt_tokens = (int*)bt_calloc(max_tokens, sizeof(int));
    int n_prompt = bt_encode(tok, prompt, prompt_tokens, max_tokens, 1);
    if (n_prompt <= 0) {
        fprintf(stderr, "biturbo: failed to encode prompt\n");
        free(prompt_tokens);
        return;
    }
    fprintf(stderr, "biturbo: %d prompt tokens\n", n_prompt);

    int token = prompt_tokens[0];
    int prev_token = 0;
    struct timespec ts_start;
    int gen_count = 0;

    for (int pos = 0; pos < max_tokens; pos++) {
        bt_forward(model, token, pos);

        int next;
        if (pos < n_prompt - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = bt_sample(sampler, model->state.logits,
                             model->config.vocab_size);
            if (gen_count == 0) clock_gettime(CLOCK_MONOTONIC, &ts_start);
            gen_count++;
        }

        if (next == tok->eos_id || next == tok->eot_id) break;

        if (pos >= n_prompt - 1) {
            const char* piece = bt_decode(tok, prev_token, next);
            printf("%s", piece);
            fflush(stdout);
        }

        prev_token = token;
        token = next;
    }
    printf("\n");

    if (gen_count > 1) {
        struct timespec ts_end;
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        double elapsed = (double)(ts_end.tv_sec - ts_start.tv_sec)
                       + (double)(ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
        fprintf(stderr, "biturbo: %d tokens in %.2fs (%.1f tok/s)\n",
                gen_count, elapsed, (double)gen_count / elapsed);
    }
    free(prompt_tokens);
}
