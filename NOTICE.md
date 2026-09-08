# Notices

The container images do not include GLM model weights. The vendor base supplies the pinned CUDA/PyTorch dependency stack; `stack.lock.json` records the SGLang and FlashInfer source inputs and integration patches.

`zai-org/GLM-5.3-Flash` is released by Z.ai under the **MIT License**. The separately distributed [quantized checkpoint](https://huggingface.co/ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO) is derived from that model and retains its license. It uses W4A16 NVFP4 routed experts with K=32 group scales and FP8 weight-only storage for eligible projections. See [QUANTIZATION.md](QUANTIZATION.md) for the complete precision selection and reproduction instructions.

SGLang is Apache-2.0. FlashInfer is Apache-2.0. The `lmsysorg/sglang` base image
carries its own upstream notices.
