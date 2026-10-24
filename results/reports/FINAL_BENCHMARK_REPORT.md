# LocalLLM execution and benchmark report

Updated 2026-10-25 (Asia/Kolkata). Values below come from local artifacts; missing stages remain pending.

## Original PDF requirements

The three-page LocalLLM PS.pdf requires one selected problem, SGLang as the core inference/serving engine, and CuratorKIT, AlignTune and AuditKIT across curation, fine-tuning and evaluation. PS4 (low-resource and multilingual public-facing language assistance) is selected. The PDF sets no numeric accuracy, BLEU, latency or throughput thresholds. The repository's 7B, 10+ language and benchmark-superiority goals are additional project targets.

Requirement assessment: the selected PS4 and required toolchain have executable local evidence. CuratorKIT generation uses SGLang, AlignTune uses TRL for SFT/DPO, and AuditKIT evaluates the SGLang endpoint including six safety smoke probes. PS4 also asks for improvement in underserved languages: completion of training does not satisfy that outcome by itself. Pilot regressions are evidence against an improvement claim, and the small samples cannot establish representative public-facing quality. Independent multilingual safety evaluation remains incomplete. The project is therefore not established as submission-ready on quality grounds.

## Current machine

Windows with Ubuntu WSL2; RTX 4070 Laptop GPU with 8188 MiB VRAM. Python 3.12.3. Training and serving environments are isolated under D:/local-llm-runtime. This supersedes the earlier Nix/RTX 4050 handoff; earlier ignored model artifacts were not present in this checkout.

## Data and model provenance

- `model_download.json`: {"model": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "path": "/mnt/d/local-llm-runtime/models/Qwen2.5-1.5B-Instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "files": {"LICENSE": 11343, "README.md": 4917, "config.json": 660, "generation_config.json": 242, "merges.txt": 1671839, "model.safetensors": 3087467144, "tokenizer.json": 7031645, "tokenizer_config.json": 7305, "vocab.json": 2776833}}

- `data_provenance.json`: {"dataset": "CohereLabs/aya_dataset", "revision": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2", "license": "apache-2.0", "acquisition": "HF dataset viewer filtered by language_code; at most 100 rows examined per language", "counts": {"train": 216, "validation": 36, "test": 36}, "train_languages": {"hi": 24, "te": 24, "bn": 24, "kn": 24, "mr": 24, "ml": 24, "pa": 24, "gu": 24, "ta": 24}, "failures": {"ori": "Insufficient usable ori rows: 0"}, "limitations": "Small convenience sample, not a representative benchmark. General instructions, not verified public-service advice. No Hinglish/Banglish coverage from Aya."}

- `flores_provenance.json`: {"dataset": "original FLORES-200 (not FLORES+)", "url": "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz", "archive_sha256": "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6", "license": "CC-BY-SA-4.0", "split": "devtest", "count": 50, "per_language": 5, "selection": "First five aligned devtest rows, fixed before model evaluation", "limitations": "Small pilot only; not full-benchmark performance. Never used for training."}

## CuratorKIT cleaning

Executed CuratorKIT 0.1.22: {"JSONLReader": {"output_count": 216, "rejected_count": 0}, "SchemaGate": {"input_count": 216, "output_count": 216, "probe_recovered": 0, "rejected_count": 0}, "TextCleaner": {"input_count": 216, "output_count": 216}, "ExactDeduplicator": {"input_count": 216, "output_count": 216}, "MessagesExporter": {"exported_count": 216}}. Manifest, checksums and provenance are in data/curated/cleaned_sft. Schema/cleaning/deduplication are not independent factual or safety review.

## Training

CuratorKIT generated QA, multi-turn and preference candidates through SGLang. Its grounding judge accepted two factual errors (Bengali triangle geometry and Tamil dengue interpreted as a wedding). Source-bound agent review excludes both; raw generations and review hashes are preserved. Seven preferences anchor chosen answers to the repository's handwritten references. Final SFT data contain 220 conversations. This review is not independent expert annotation.

### SFT

{"status": "completed", "toolkit": "aligntune", "versions": {"aligntune": "0.1.28", "torch": "2.11.0", "transformers": "5.20.2", "trl": "1.7.1", "peft": "0.21.2", "bitsandbytes": "0.50.2"}, "model": "Qwen/Qwen2.5-1.5B-Instruct", "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "dataset_count": 220, "dataset_sha256": "e504f9724bb205fa69bd994b6c30843b6a3c11a38195fe1d8c4662ddeb37d975", "duration_seconds": 137.6016662120819, "peak_allocated_vram_bytes": 2950186496, "global_steps": 55, "adapter_path": "/mnt/c/Users/Krish/local-llm/outputs/local_1_5b_sft", "limitations": "Small local proof-of-pipeline; held-out benchmark comparison required before claiming improvement.", "toolkit_provenance": "outputs/local_1_5b_sft/lexsi_provenance.json", "mean_training_loss": 1.3475896987048062}

### DPO

{"status": "completed", "toolkit": "aligntune", "dataset_count": 7, "duration_seconds": 50.69304609298706, "peak_allocated_vram_bytes": 3652478464, "global_steps": 2, "adapter_path": "/mnt/c/Users/Krish/local-llm/outputs/local_1_5b_dpo", "limitations": "Tiny synthetic preference set; model-generated preferences require native-speaker review and do not establish safety.", "toolkit_provenance": "outputs/local_1_5b_dpo/lexsi_provenance.json", "mean_training_loss": 0.6896083950996399}

## Evaluation

Aya is a convenience holdout, not a standard benchmark. FLORES-200 uses a fixed first-five-devtest-sentences pilot per language, not the full benchmark. AuditKIT sentence BLEU/ROUGE-L/chrF use 0-1 scales. SacreBLEU spBLEU (FLORES-200 tokenizer) and chrF++ use 0-100 scales; exact signatures are recorded in each JSON. These tiny samples do not establish statistical superiority.

### baseline

- aya: n=36; AuditKIT metrics={"bleu": 0.004770669185929175, "rouge_l": 0.08523267407088211, "chrf": 0.15342990020888464}; performance={"latency_ms": {"count": 36, "mean": 3189.924814055552, "min": 311.5612139999939, "max": 3970.1856770000177, "p50": 3285.557103999963, "p95": 3728.891888000021, "p99": 3775.7806549999486, "std": 714.646462522038}, "throughput": {"rps": 0.3134798027104836, "total_requests": 36, "total_time_ms": 114839.93446699993, "output_tokens_per_sec": 37.4695441962003, "total_tokens_per_sec": 83.82972390816182}}.
  - bn: {"count": 4, "auditkit_metrics": {"bleu": 0.019517041809299016, "rouge_l": 0.10044017607042817, "chrf": 0.1839182076664454}}
  - gu: {"count": 4, "auditkit_metrics": {"bleu": 2.5052127519775745e-10, "rouge_l": 0.035663199729345195, "chrf": 0.0775456967934137}}
  - hi: {"count": 4, "auditkit_metrics": {"bleu": 0.022966715709202848, "rouge_l": 0.21346443154953793, "chrf": 0.2475601475514659}}
  - kn: {"count": 4, "auditkit_metrics": {"bleu": 6.318689928661352e-08, "rouge_l": 0.10062111801242235, "chrf": 0.20856760091777313}}
  - ml: {"count": 4, "auditkit_metrics": {"bleu": 1.5062007065973263e-10, "rouge_l": 0.07594007050528791, "chrf": 0.1599559413808716}}
  - mr: {"count": 4, "auditkit_metrics": {"bleu": 1.0682865325932473e-07, "rouge_l": 0.07603265505056658, "chrf": 0.1346887680903957}}
  - pa: {"count": 4, "auditkit_metrics": {"bleu": 0.0004520945193952351, "rouge_l": 0.11817541613316261, "chrf": 0.15246311885917668}}
  - ta: {"count": 4, "auditkit_metrics": {"bleu": 2.1332489654516148e-10, "rouge_l": 0.03732303732303732, "chrf": 0.14161598030546244}}
  - te: {"count": 4, "auditkit_metrics": {"bleu": 5.446683917762123e-12, "rouge_l": 0.009433962264150943, "chrf": 0.07455364031495726}}
- flores: n=50; AuditKIT metrics={"bleu": 0.003550275454735971, "rouge_l": 0.05715495612802418, "chrf": 0.1631051492850159}; performance={"latency_ms": {"count": 50, "mean": 3309.912824520004, "min": 1777.9427149999947, "max": 4085.862169000052, "p50": 3345.3538029999663, "p95": 3788.9697499999784, "p99": 3919.6463040000253, "std": 414.5193740485348}, "throughput": {"rps": 0.30211413146250815, "total_requests": 50, "total_time_ms": 165500.36821499997, "output_tokens_per_sec": 37.1358690393715, "total_tokens_per_sec": 62.030073471882176}}.
  - ben_Beng: {"count": 5, "auditkit_metrics": {"bleu": 0.03548432901464226, "rouge_l": 0.08490631973390593, "chrf": 0.21630031430260113}, "spBLEU": 2.4550182910951786, "chrF++": 14.578618800153375, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - guj_Gujr: {"count": 5, "auditkit_metrics": {"bleu": 1.1834382108157123e-10, "rouge_l": 0.03358454106280193, "chrf": 0.14051078233783607}, "spBLEU": 0.25988579509535437, "chrF++": 7.851266535238441, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - hin_Deva: {"count": 5, "auditkit_metrics": {"bleu": 1.8424711225169723e-05, "rouge_l": 0.2023565388100272, "chrf": 0.2609979341354381}, "spBLEU": 3.94531123345277, "chrF++": 16.814304166410132, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - kan_Knda: {"count": 5, "auditkit_metrics": {"bleu": 1.243680933991255e-10, "rouge_l": 0.04454312398553044, "chrf": 0.1627147573539917}, "spBLEU": 0.41833892857038835, "chrF++": 9.986360538191317, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - mal_Mlym: {"count": 5, "auditkit_metrics": {"bleu": 9.614860520448452e-11, "rouge_l": 0.014285714285714285, "chrf": 0.11921637800861487}, "spBLEU": 0.1237457591796625, "chrF++": 7.43238178320758, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - mar_Deva: {"count": 5, "auditkit_metrics": {"bleu": 1.7695238332488768e-10, "rouge_l": 0.06615902615902616, "chrf": 0.20867889780451407}, "spBLEU": 0.5432917971791478, "chrF++": 13.200426734382395, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - ory_Orya: {"count": 5, "auditkit_metrics": {"bleu": 5.052526829010321e-13, "rouge_l": 0.0, "chrf": 0.01892963813715759}, "spBLEU": 0.20275553229433557, "chrF++": 0.661143945760241, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - pan_Guru: {"count": 5, "auditkit_metrics": {"bleu": 1.2579496198549688e-10, "rouge_l": 0.07728100470957613, "chrf": 0.16531363007233554}, "spBLEU": 0.248316942463764, "chrF++": 8.691450185023792, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - tam_Taml: {"count": 5, "auditkit_metrics": {"bleu": 1.352513091227895e-10, "rouge_l": 0.03843329253365973, "chrf": 0.195499698844502}, "spBLEU": 0.18642462984594083, "chrF++": 11.40929930836196, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - tel_Telu: {"count": 5, "auditkit_metrics": {"bleu": 4.412785188514549e-11, "rouge_l": 0.01, "chrf": 0.1428894618531681}, "spBLEU": 0.20934727154170008, "chrF++": 8.136211734637243, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}

Safety smoke: 6 probes; keyword score={"referral_keyword_screen": 0.5}. Lexical referral/refusal markers are a smoke screen only. They can false-pass unsafe responses and false-fail safe paraphrases. Human/native-speaker review, toxicity/bias and factual hallucination audits remain required.

### sft

- aya: n=36; AuditKIT metrics={"bleu": 0.004232002403791035, "rouge_l": 0.073435753921906, "chrf": 0.15085276284238802}; performance={"latency_ms": {"count": 36, "mean": 2872.3579424166646, "min": 404.48474199999396, "max": 4100.826188999974, "p50": 3242.0209599999907, "p95": 3833.901290999989, "p99": 3976.969101000009, "std": 1051.2751502987849}, "throughput": {"rps": 0.3481390415528873, "total_requests": 36, "total_time_ms": 103406.96016000002, "output_tokens_per_sec": 36.61262253664531, "total_tokens_per_sec": 88.09851857074452}}.
  - bn: {"count": 4, "auditkit_metrics": {"bleu": 0.017515775418784296, "rouge_l": 0.15358778943861404, "chrf": 0.22062267138245953}}
  - gu: {"count": 4, "auditkit_metrics": {"bleu": 1.807435204424939e-10, "rouge_l": 0.030584363433200645, "chrf": 0.0864029970563349}}
  - hi: {"count": 4, "auditkit_metrics": {"bleu": 0.020545202579661654, "rouge_l": 0.21543927648578812, "chrf": 0.23422157534756186}}
  - kn: {"count": 4, "auditkit_metrics": {"bleu": 2.804170474701935e-08, "rouge_l": 0.05518018018018018, "chrf": 0.15884474813409247}}
  - ml: {"count": 4, "auditkit_metrics": {"bleu": 7.486327007354485e-12, "rouge_l": 0.042270531400966184, "chrf": 0.23826626470701284}}
  - mr: {"count": 4, "auditkit_metrics": {"bleu": 7.269953296673205e-12, "rouge_l": 0.007067137809187279, "chrf": 0.05612169831130947}}
  - pa: {"count": 4, "auditkit_metrics": {"bleu": 2.7015294098928355e-05, "rouge_l": 0.13543789050090013, "chrf": 0.15425206532315075}}
  - ta: {"count": 4, "auditkit_metrics": {"bleu": 9.79579062855645e-11, "rouge_l": 0.013157894736842105, "chrf": 0.12761338272315165}}
  - te: {"count": 4, "auditkit_metrics": {"bleu": 6.411981905132431e-12, "rouge_l": 0.008196721311475409, "chrf": 0.08132946259641854}}
- flores: n=50; AuditKIT metrics={"bleu": 0.0036898534055534614, "rouge_l": 0.05142339199205704, "chrf": 0.14652393330637706}; performance={"latency_ms": {"count": 50, "mean": 3309.764265439997, "min": 1301.6164989999766, "max": 5550.912786999959, "p50": 3264.9090909999927, "p95": 3601.6484000000446, "p99": 5031.306577999999, "std": 539.6456174912479}, "throughput": {"rps": 0.302130864841909, "total_requests": 50, "total_time_ms": 165491.20205300002, "output_tokens_per_sec": 37.66967622848921, "total_tokens_per_sec": 62.56525949146251}}.
  - ben_Beng: {"count": 5, "auditkit_metrics": {"bleu": 0.036878503978028235, "rouge_l": 0.10848108474158895, "chrf": 0.22481303633529395}, "spBLEU": 2.430961936044117, "chrF++": 14.885005549658015, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - guj_Gujr: {"count": 5, "auditkit_metrics": {"bleu": 3.3191913157184663e-13, "rouge_l": 0.0, "chrf": 0.11657916369614918}, "spBLEU": 0.11330244224773037, "chrF++": 6.1955276130021035, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - hin_Deva: {"count": 5, "auditkit_metrics": {"bleu": 2.0017770435104997e-05, "rouge_l": 0.20092265547589902, "chrf": 0.28034186240356107}, "spBLEU": 3.9846916711185143, "chrF++": 18.01495170490586, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - kan_Knda: {"count": 5, "auditkit_metrics": {"bleu": 1.6084417880376882e-10, "rouge_l": 0.020460358056265983, "chrf": 0.11844006853217406}, "spBLEU": 0.1863578643606558, "chrF++": 6.66761162948679, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - mal_Mlym: {"count": 5, "auditkit_metrics": {"bleu": 6.4108199244342e-13, "rouge_l": 0.0, "chrf": 0.09349748222452595}, "spBLEU": 0.15328709741894142, "chrF++": 5.192532691965085, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - mar_Deva: {"count": 5, "auditkit_metrics": {"bleu": 2.2381542634487436e-10, "rouge_l": 0.06966539407748815, "chrf": 0.16802196262787036}, "spBLEU": 0.6366772480778162, "chrF++": 11.73359039343951, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - ory_Orya: {"count": 5, "auditkit_metrics": {"bleu": 3.5294914411143717e-13, "rouge_l": 0.0, "chrf": 0.016358130324107698}, "spBLEU": 0.1658596270243679, "chrF++": 0.5859294347765569, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - pan_Guru: {"count": 5, "auditkit_metrics": {"bleu": 1.1749709433689171e-08, "rouge_l": 0.0811517959903809, "chrf": 0.16883569944143112}, "spBLEU": 0.40581246424193895, "chrF++": 9.07623996868805, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - tam_Taml: {"count": 5, "auditkit_metrics": {"bleu": 8.219302780648546e-11, "rouge_l": 0.02105263157894737, "chrf": 0.14951362520949568}, "spBLEU": 0.09692254224942372, "chrF++": 8.279998214038606, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - tel_Telu: {"count": 5, "auditkit_metrics": {"bleu": 8.918326005728238e-11, "rouge_l": 0.0125, "chrf": 0.1288383022691615}, "spBLEU": 0.08079826533274101, "chrF++": 6.917025608332886, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}

Safety smoke: 6 probes; keyword score={"referral_keyword_screen": 0.3333333333333333}. Lexical referral/refusal markers are a smoke screen only. They can false-pass unsafe responses and false-fail safe paraphrases. Human/native-speaker review, toxicity/bias and factual hallucination audits remain required.

### dpo

- aya: n=36; AuditKIT metrics={"bleu": 0.004406251865816922, "rouge_l": 0.076207603136708, "chrf": 0.1530157739319078}; performance={"latency_ms": {"count": 36, "mean": 2813.084903861134, "min": 331.0557229999631, "max": 3930.6079129999034, "p50": 3297.6723280000897, "p95": 3554.974909000066, "p99": 3610.9822910000275, "std": 1025.1579558243411}, "throughput": {"rps": 0.35547394125943516, "total_requests": 36, "total_time_ms": 101273.24628200007, "output_tokens_per_sec": 36.890295681812496, "total_tokens_per_sec": 89.46094188362451}}.
  - bn: {"count": 4, "auditkit_metrics": {"bleu": 0.017515777684477377, "rouge_l": 0.15312525474851227, "chrf": 0.22118198369565578}}
  - gu: {"count": 4, "auditkit_metrics": {"bleu": 6.196615318097778e-10, "rouge_l": 0.04090718625602347, "chrf": 0.0969315324211117}}
  - hi: {"count": 4, "auditkit_metrics": {"bleu": 0.022113445052738143, "rouge_l": 0.21760755584284996, "chrf": 0.2521616642419754}}
  - kn: {"count": 4, "auditkit_metrics": {"bleu": 2.804170474701935e-08, "rouge_l": 0.05518018018018018, "chrf": 0.15929424832863842}}
  - ml: {"count": 4, "auditkit_metrics": {"bleu": 7.486327007354485e-12, "rouge_l": 0.042270531400966184, "chrf": 0.23826626470701284}}
  - mr: {"count": 4, "auditkit_metrics": {"bleu": 5.010236399729374e-13, "rouge_l": 0.003731343283582089, "chrf": 0.03402775235876242}}
  - pa: {"count": 4, "auditkit_metrics": {"bleu": 2.7015294098928355e-05, "rouge_l": 0.13543789050090013, "chrf": 0.15425206532315075}}
  - ta: {"count": 4, "auditkit_metrics": {"bleu": 8.526783440887658e-11, "rouge_l": 0.029411764705882353, "chrf": 0.14274266606484845}}
  - te: {"count": 4, "auditkit_metrics": {"bleu": 6.41639243222648e-12, "rouge_l": 0.008196721311475409, "chrf": 0.07828378824601434}}
- flores: n=50; AuditKIT metrics={"bleu": 0.005011858103924721, "rouge_l": 0.05716594245014917, "chrf": 0.14985337383345118}; performance={"latency_ms": {"count": 50, "mean": 3271.4896404799856, "min": 1300.0842569999804, "max": 3783.2436399999096, "p50": 3339.9601930000244, "p95": 3449.1922709999017, "p99": 3643.4290600000168, "std": 349.3602754371598}, "throughput": {"rps": 0.3056626508468365, "total_requests": 50, "total_time_ms": 163579.03022000004, "output_tokens_per_sec": 38.13447231965133, "total_tokens_per_sec": 63.32107474943066}}.
  - ben_Beng: {"count": 5, "auditkit_metrics": {"bleu": 0.03687850399261536, "rouge_l": 0.1150337912942955, "chrf": 0.22080510035417894}, "spBLEU": 2.3873226441167277, "chrF++": 14.483224471712875, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - guj_Gujr: {"count": 5, "auditkit_metrics": {"bleu": 9.88965497390104e-11, "rouge_l": 0.0404040404040404, "chrf": 0.13018213125584802}, "spBLEU": 0.15088787416945237, "chrF++": 7.20088625739227, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - hin_Deva: {"count": 5, "auditkit_metrics": {"bleu": 0.01324005800572615, "rouge_l": 0.2061708370713306, "chrf": 0.26151578097917605}, "spBLEU": 4.183820520828326, "chrF++": 17.023250131168567, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - kan_Knda: {"count": 5, "auditkit_metrics": {"bleu": 8.352351921853858e-11, "rouge_l": 0.01212121212121212, "chrf": 0.11071206924368335}, "spBLEU": 0.11593683349919413, "chrF++": 6.025553398020946, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - mal_Mlym: {"count": 5, "auditkit_metrics": {"bleu": 5.927301086720204e-13, "rouge_l": 0.0, "chrf": 0.11138795520800013}, "spBLEU": 0.1949923315179128, "chrF++": 6.368378488338496, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - mar_Deva: {"count": 5, "auditkit_metrics": {"bleu": 1.6963400121284325e-10, "rouge_l": 0.051976276388370456, "chrf": 0.1638849452545557}, "spBLEU": 0.5378511387446586, "chrF++": 11.06745197046737, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - ory_Orya: {"count": 5, "auditkit_metrics": {"bleu": 3.5294914411143717e-13, "rouge_l": 0.0, "chrf": 0.016189198053159902}, "spBLEU": 0.18176397958356766, "chrF++": 0.585613509318144, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - pan_Guru: {"count": 5, "auditkit_metrics": {"bleu": 1.8432029908852875e-08, "rouge_l": 0.09730132580714332, "chrf": 0.17416443738781823}, "spBLEU": 0.42716676033203466, "chrF++": 9.328189092004981, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - tam_Taml: {"count": 5, "auditkit_metrics": {"bleu": 1.3873194771762273e-10, "rouge_l": 0.025341130604288498, "chrf": 0.15717210833625503}, "spBLEU": 0.10271573115493271, "chrF++": 8.407419686529181, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}
  - tel_Telu: {"count": 5, "auditkit_metrics": {"bleu": 1.1714409863089117e-10, "rouge_l": 0.023310810810810812, "chrf": 0.15252001226183637}, "spBLEU": 0.11644450646499706, "chrF++": 8.760103659901276, "signatures": {"spBLEU": "nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0", "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"}}

Safety smoke: 6 probes; keyword score={"referral_keyword_screen": 0.3333333333333333}. Lexical referral/refusal markers are a smoke screen only. They can false-pass unsafe responses and false-fail safe paraphrases. Human/native-speaker review, toxicity/bias and factual hallucination audits remain required.

### sft minus baseline: FLORES chrF++ points

- ben_Beng: +0.3064 points.
- guj_Gujr: -1.6557 points.
- hin_Deva: +1.2006 points.
- kan_Knda: -3.3187 points.
- mal_Mlym: -2.2398 points.
- mar_Deva: -1.4668 points.
- ory_Orya: -0.0752 points.
- pan_Guru: +0.3848 points.
- tam_Taml: -3.1293 points.
- tel_Telu: -1.2192 points.

Unweighted ten-language mean chrF++: baseline 9.8761, sft 8.7548, difference -1.1213 points. This five-sentence-per-language pilot is descriptive only.

### dpo minus baseline: FLORES chrF++ points

- ben_Beng: -0.0954 points.
- guj_Gujr: -0.6504 points.
- hin_Deva: +0.2089 points.
- kan_Knda: -3.9608 points.
- mal_Mlym: -1.0640 points.
- mar_Deva: -2.1330 points.
- ory_Orya: -0.0755 points.
- pan_Guru: +0.6367 points.
- tam_Taml: -3.0019 points.
- tel_Telu: +0.6239 points.

Unweighted ten-language mean chrF++: baseline 9.8761, dpo 8.9250, difference -0.9511 points. This five-sentence-per-language pilot is descriptive only.

## Final verification

- sft_export_verification: {"status": "passed", "stage": "sft", "vocabulary_identical": true, "sample_languages_checked": 12, "evaluation_prompts_checked": 86, "chat_template_token_ids_identical": true, "architecture_fields_checked": ["model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size", "rope_theta", "rms_norm_eps", "tie_word_embeddings", "torch_dtype"], "rope_theta": 1000000.0}

- dpo_export_verification: {"status": "passed", "stage": "dpo", "vocabulary_identical": true, "sample_languages_checked": 12, "evaluation_prompts_checked": 86, "chat_template_token_ids_identical": true, "architecture_fields_checked": ["model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size", "rope_theta", "rms_norm_eps", "tie_word_embeddings", "torch_dtype"], "rope_theta": 1000000.0}

- gui_verification_before_history_fix: {"status": "passed", "url": "http://127.0.0.1:7860", "loaded_model_path": "/mnt/c/Users/Krish/local-llm/outputs/local_1_5b_dpo_merged", "checks": ["HTTP page", "real queued streaming", "Unicode input preserved", "Hindi-script response", "follow-up history", "empty message", "clear"], "stream_updates": 34, "prompt": "नमस्ते! दो और दो कितने होते हैं? केवल एक छोटा वाक्य लिखें।", "response": "दो और दो होते हैं, तीन छोटा वाक्य लिखें।", "followup_response": "[{'text': 'दो और दो = 4', 'type': 'number'}]", "visual_browser_check": "blocked: no browser connector; computer-use tool rejected Chrome access as not approved", "limitations": "Functional API smoke test; not independent language-quality or visual-layout verification."}

- gui_verification: {"status": "passed", "url": "http://127.0.0.1:7860", "loaded_model_path": "/mnt/c/Users/Krish/local-llm/outputs/local_1_5b_dpo_merged", "checks": ["HTTP page", "real queued streaming", "Unicode input preserved", "Hindi-script response", "follow-up history", "empty message", "clear"], "stream_updates": 32, "prompt": "नमस्ते! दो और दो कितने होते हैं? केवल एक छोटा वाक्य लिखें।", "response": "दो और दो होते हैं, तीन कितने होते हैं?", "followup_response": "2 और 2 होते हैं, तीन कितने होते हैं?", "arithmetic_answer_marker_present": false, "quality_limit": "Arithmetic marker presence is a limited diagnostic, not a factual correctness score. Functional success must not be read as model-quality success. The initial pre-fix GUI responses are preserved in gui_verification_before_history_fix.json.", "visual_browser_check": "blocked: no browser connector; computer-use tool rejected Chrome access as not approved", "limitations": "Functional API smoke test; not independent language-quality or visual-layout verification."}

- final_verification: {"status": "passed", "checks": {"pytest": {"exit_code": 0, "output": ".......................                                                  [100%]\n23 passed in 6.69s"}, "compile": {"exit_code": 0, "output": ""}, "python_lint": {"exit_code": 0, "output": "All checks passed!"}, "setup": {"exit_code": 0, "output": "Python: 3.12.3\nOK directory: data\nOK directory: configs\nOK directory: scripts\nOK directory: src/gui\nOK directory: outputs\nOK directory: results\nOK directory: memory\nOK dependency: yaml 6.0.3\nOK dependency: requests 2.34.2\nGPU: {'available': True, 'device_count': 1, 'device_name': 'NVIDIA GeForce RTX 4070 Laptop GPU', 'total_memory_gb': 8.59}\nOK package: curatorkit 0.1.22\nOK package: aligntune 0.1.28\nOK package: auditkit 1.1.3\nOK package: gradio 6.30.0\nOK package: transformers 5.20.2\nOK package: trl 1.7.1\nOK package: peft 0.21.2\nOK package: bitsandbytes 0.50.2\n\nSetup structure is valid."}, "jsonl_files": {"exit_code": 0, "output": "data/raw\n  data/raw/aya_cache/ben.jsonl: 32\n  data/raw/aya_cache/guj.jsonl: 32\n  data/raw/aya_cache/hin.jsonl: 32\n  data/raw/aya_cache/kan.jsonl: 32\n  data/raw/aya_cache/mal.jsonl: 32\n  data/raw/aya_cache/mar.jsonl: 32\n  data/raw/aya_cache/pan.jsonl: 32\n  data/raw/aya_cache/tam.jsonl: 32\n  data/raw/aya_cache/tel.jsonl: 32\n  data/raw/seeds/all_seeds.jsonl: 6\n  data/raw/seeds/preference_seeds.jsonl: 8\n  data/raw/seeds/seeds_agriculture.jsonl: 1\n  data/raw/seeds/seeds_education.jsonl: 2\n  data/raw/seeds/seeds_healthcare.jsonl: 1\n  data/raw/seeds/seeds_public_services.jsonl: 2\n  data/raw/sft/indic_aya_train.jsonl: 216\n\ndata/curated\n  data/curated/cleaned_sft/rejected.jsonl: 0\n  data/curated/cleaned_sft/sft_messages.jsonl: 216\n  data/curated/final_sft_train.jsonl: 220\n  data/curated/multiturn_generation/rejected.jsonl: 2\n  data/curated/multiturn_generation/sft_messages.jsonl: 4\n  data/curated/preference_generation/dpo.jsonl: 8\n  data/curated/preference_generation/rejected.jsonl: 0\n  data/curated/qa_generation/rejected.jsonl: 8\n  data/curated/qa_generation/sft_messages.jsonl: 2\n  data/curated/reviewed_preferences/dpo.jsonl: 7\n  data/curated/reviewed_preferences/rejected.jsonl: 0\n  data/curated/sft_messages_train.jsonl: 220\n\ndata/preference\n  data/preference/multilingual_pref.jsonl: 7\n  data/preference/source_anchored_pairs.jsonl: 7\n\ndata/eval\n  data/eval/aya_test.jsonl: 36\n  data/eval/aya_validation.jsonl: 36\n  data/eval/flores200.jsonl: 50\n\nSerialized rows across sources, caches and exports (not unique training examples): 1347\nRow language metadata across those files: {'ben_Beng': 5, 'bn': 115, 'gu': 114, 'guj_Gujr': 5, 'hi': 118, 'hi-en': 4, 'hin_Deva': 5, 'kan_Knda': 5, 'kn': 112, 'mal_Mlym': 5, 'mar_Deva': 5, 'ml': 112, 'mr': 114, 'ory_Orya': 5, 'pa': 114, 'pan_Guru': 5, 'ta': 116, 'tam_Taml': 5, 'te': 116, 'tel_Telu': 5, 'unknown': 262}"}, "dependencies_train": {"exit_code": 0, "output": "Using Python 3.12.3 environment at: /mnt/d/local-llm-runtime/train\nChecked 291 packages in 913ms\nAll installed packages are compatible"}, "dependencies_serve": {"exit_code": 0, "output": "Using Python 3.12.3 environment at: /mnt/d/local-llm-runtime/serve\nChecked 174 packages in 569ms\nAll installed packages are compatible"}, "sft_training": {"global_steps": 55, "dataset_count": 220, "validated_dataset_sha256": "e504f9724bb205fa69bd994b6c30843b6a3c11a38195fe1d8c4662ddeb37d975"}, "dpo_training": {"global_steps": 2, "dataset_count": 7, "validated_dataset_sha256": "5b9abffe878e47ef65221c1adef3fb80008c1397e8bf7473210b230826e9ac2b"}, "baseline_evaluation": {"quality_predictions": 86, "safety_predictions": 6, "checkpoint_identity": {"loaded_model_path": "/mnt/d/local-llm-runtime/models/Qwen2.5-1.5B-Instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "weights_sha256": {"model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"}}}, "sft_evaluation": {"quality_predictions": 86, "safety_predictions": 6, "checkpoint_identity": {"loaded_model_path": "/mnt/c/Users/Krish/local-llm/outputs/local_1_5b_sft_merged", "weights_sha256": {"model.safetensors": "9c02f4be6e93088016ae2b4d59439f9ac3e313a7f30bd2fd1139870b417466e4"}}}, "dpo_evaluation": {"quality_predictions": 86, "safety_predictions": 6, "checkpoint_identity": {"loaded_model_path": "/mnt/c/Users/Krish/local-llm/outputs/local_1_5b_dpo_merged", "weights_sha256": {"model.safetensors": "a1deb30b310994759b2e414fa2f11905f5fa37963e3e899f3406f75c95aa5ff0"}}}, "data_leakage": "No normalized SFT/DPO user prompt matches the 86 frozen quality-evaluation prompts or 36 validation prompts"}, "limitations": "Technical local-pipeline checks. Full benchmarks, independent safety/language review and 7B training are not completed. Browser visual verification was blocked by computer-use access."}

The live GUI's functional API checks passed, but the model failed the simple Hindi 2+2 smoke question: it did not state four and produced an irrelevant question. This failure persisted after correcting Gradio 6 history text-block handling. The failed responses are preserved, and GUI functional success is not model-quality success.

AuditKIT's API cache originally reused baseline predictions when different checkpoints shared local-llm. Evaluation now fingerprints actual safetensors contents and loaded model path, and a regression test checks separation even when weights change at the same path. The invalid reused SFT results were replaced by fresh requests; model-specific identity is recorded in the final JSON artifacts.

## Remaining limits

- The 7B production training target requires a larger GPU; only the explicitly named 1.5B local profile is appropriate here.
- Full FLORES-200, MMLU and IndicXTREME evaluation and representative native-speaker public-service review are not replaced by these pilots.
- Odia and meaningful code-mixed training coverage remain incomplete in Aya. A language appearing in a GUI dropdown does not demonstrate model competence.
- Synthetic model judges and lexical safety probes are limited evidence. No clinical, governmental, toxicity, bias or hallucination safety certification is claimed.
- Negative benchmark deltas must be preserved; an executed pipeline does not prove an improved model.

## Evidence and reproduction

See README.md for current WSL commands. Inspect results/reports/*.json, results/benchmarks/<stage>/, results/safety/<stage>/, outputs/local_1_5b_* and memory/10_remaining_work_handoff.md. A stage is complete only when its artifact and successful verification exist.
