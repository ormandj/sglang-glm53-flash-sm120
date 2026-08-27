# Notices

This repository and its container build recipe do not redistribute GLM model
weights. The build starts from a public upstream SGLang image and adds an
SM120-targeted FlashInfer build; no checkpoint is baked in.

`zai-org/GLM-5.3-Flash` is released by Z.ai under the **MIT License**. Obtain it
separately and comply with that license. The served artifact in this homelab is
an MXFP4A16 quantization of that checkpoint produced locally; it is a derived
work of an MIT-licensed model and is not published here.

SGLang is Apache-2.0. FlashInfer is Apache-2.0. The `lmsysorg/sglang` base image
carries its own upstream notices.
