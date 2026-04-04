/*
 * main.c — biturbo CLI runner
 *
 * Usage: biturbo <model.gguf> [options]
 *   -p <prompt>      Input prompt (default: "Hello")
 *   -n <count>       Max tokens to generate (default: 256)
 *   -t <temp>        Temperature (default: 0.8, 0.0 = greedy)
 *   -k <top_p>       Top-p nucleus sampling (default: 0.9)
 *   -s <seed>        RNG seed (default: time-based)
 */

#include "biturbo.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char* prog) {
    fprintf(stderr,
        "biturbo — BitNet 1.58-bit inference with TurboQuant INT4 KV cache\n\n"
        "Usage: %s <model.gguf> [options]\n\n"
        "Options:\n"
        "  -p <prompt>   Input prompt (default: \"Hello\")\n"
        "  -n <count>    Max tokens to generate (default: 256)\n"
        "  -t <temp>     Temperature, 0.0 = greedy (default: 0.8)\n"
        "  -k <top_p>    Top-p nucleus sampling (default: 0.9)\n"
        "  -s <seed>     RNG seed (default: time-based)\n"
        "  -h            Show this help\n\n"
        "Loads GGUF models with I2_S (1.58-bit ternary) weights.\n"
        "KV cache quantized to INT4 via TurboQuant uniform scheme.\n",
        prog);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    const char* model_path = NULL;
    const char* prompt = "Hello";
    int max_tokens = 256;
    float temperature = 0.8f;
    float top_p = 0.9f;
    uint64_t seed = 0;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') { model_path = argv[i]; continue; }
        if (strcmp(argv[i], "-h") == 0) { usage(argv[0]); return 0; }
        if (i + 1 >= argc) {
            fprintf(stderr, "error: %s needs argument\n", argv[i]);
            return 1;
        }
        switch (argv[i][1]) {
            case 'p': prompt = argv[++i]; break;
            case 'n': max_tokens = atoi(argv[++i]); break;
            case 't': temperature = (float)atof(argv[++i]); break;
            case 'k': top_p = (float)atof(argv[++i]); break;
            case 's': seed = (uint64_t)atoll(argv[++i]); break;
            default:
                fprintf(stderr, "unknown option '%s'\n", argv[i]);
                return 1;
        }
    }

    if (!model_path) { usage(argv[0]); return 1; }

    bt_model_t model;
    if (bt_load_model(&model, model_path) != 0) return 1;

    bt_sampler_t sampler;
    bt_sampler_init(&sampler, temperature, top_p, seed);

    bt_config_t* cfg = &model.config;
    fprintf(stderr, "biturbo: %d layers, %d/%d heads, head_dim=%d, "
            "KV cache INT4 (%d blk/head)\n",
            cfg->n_layers, cfg->n_heads, cfg->n_kv_heads,
            BT_HEAD_DIM(cfg),
            (BT_HEAD_DIM(cfg) + BT_QK - 1) / BT_QK);

    bt_generate(&model, &sampler, prompt, max_tokens);
    bt_free_model(&model);
    return 0;
}
