# Upstream PR maintenance after v0.3.2

These checks validate isolated repaired PR snapshots, separately from the immutable v0.3.2 image. The tests ran in the qualification container's isolated test pod with each PR's source staged on SGLang main `28457f0dca`; they did not run inside the active serving pod. The final review branches are based on `2c05ed4e77`. Its two intervening commits change unrelated ROCm/NPU files, and each repaired diff is byte-identical across that base refresh. Commit metadata corrections preserve source trees.

| PR | Author validation |
|---|---|
| #37168 | 11 CPU tests, two CUDA ownership tests and 25 MHC tests passed; unsupported DeepGEMM paged-MQA case deselected |
| #37538 | 38 CPU tests passed, including the #37169 dependency |
| #37625 | 12 KPool tests and 38 subtests, 16 focused JIT tests and 14 source-built CUDA AOT cases passed; ROCm AOT not run |
| #35954 | 232 CPU tests and 26 subtests passed |
| #32815 | 91 CPU tests and 29 subtests passed, one skip |
| #32686 | Five CPU tests passed |

`final-commits.json` records old and new PR heads, base commits and the unchanged-patch check; the patches and test logs are retained here. The staged snapshots for the first three PRs were `5cf1951147`, `0d5b3e7456` and `e46bd14a91`. The other three were staged, uncommitted patches when tested. Tests used CUDA 13 on RTX PRO 6000 Blackwell (SM120); CPU suites used a hidden GPU index. These checks do not repeat historical end-to-end performance or accuracy experiments, qualify other hardware, or alter the released image. Existing performance tables in the older PRs remain reports of their originally identified sources.
