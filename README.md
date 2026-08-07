# Introduction 
TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project. 

# Getting Started
TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

# Build and Test
TODO: Describe and show how to build your code and run the tests. 

## Building a quality-prediction dataset

`qp-build-dataset` accepts both HTRflow layouts used by Swedish Lion 26:

* pages whose top-level `regions` are text lines; and
* pages containing text regions, with text lines in their nested `regions`.

Line-only pages retain their lines directly and have zero text regions; the
loader does not synthesize one region per line.

Select only feature groups supported by a pipeline with repeatable `--feature`
arguments (a comma-separated value is also accepted). Select target columns in
the same way with `--target`. For example, a line-only prediction can omit all
region-specific features and targets:

```bash
qp-build-dataset \
  --out-csv line-quality.csv \
  --feature htr_confidence,text,layout \
  --target target_perm_cer_strict,target_map50_line \
  --dataset sl26 ground-truth/page line_sl26
```

For region-then-line output, `--feature segmentation,regionization,layout` and
region targets such as `--target target_map50_region` are available. Run
`qp-build-dataset --help` for every feature group. `--char-lm` is needed only
when `lm` or `interaction` is selected, `--ngram-sets` only for `ngram`, and
`--bin-config` is optional.

### Swedish Lion 26: JSON-only feature inputs

The following build uses only values embedded in the pipeline JSON (geometry,
segmentation confidence, HTR confidence, and recognized text). It does **not**
load the source image, a language model, n-gram sets, or manually supplied
metadata:

```bash
GT_DIR=/path/to/corresponding/pagexml_ground_truth
PRED_DIR=/data/eva_swedish_lion_26/region_line_sl26

qp-build-dataset \
  --out-csv /data/eva_swedish_lion_26/region_line_sl26_dataset.csv \
  --dataset sl26 "$GT_DIR" "$PRED_DIR" \
  --feature segmentation,regionization,layout,htr_confidence,text \
  --target target_perm_cer_strict,target_map50_line
```

Pairing is recursive and is by filename stem: for example, `abc.json` is
paired with `abc.xml`, even when the two files are in different nested
directories. Duplicate stems within one input tree are ambiguous; only the
first discovered file is used. The output has one row per matched page and
includes identifiers, the selected prediction-time features, and the selected
ground-truth-derived targets.

Repeat `--target`, or provide a comma-separated list, to choose any target
combination. Omitting `--target` includes every target. To see the authoritative
choices without starting a build, run:

```bash
qp-build-dataset --list-targets
qp-build-dataset --list-features
```

Useful target families are:

* transcription quality: `target_perm_cer_strict`,
  `target_perm_cer_htr_only`, `target_avg_line_cer`, and `target_bow_f1`;
* missing or hallucinated content: `target_pi_missing_ratio` and
  `target_pi_halluc_ratio`;
* line segmentation: `target_map50_line`, `target_map75_line`, and the
  `target_iou*` / `target_soft_iou*` precision, recall, and F1 columns;
* region segmentation: `target_map50_region` and `target_map75_region`.

Targets are labels computed by comparing JSON predictions with PAGE XML, so
they may use ground truth; the selected feature columns do not. Avoid the
`image`, `dit`, `ngram`, `lm`, `lexicon`, and `metadata` groups for the strict
JSON-only setup above. If the PageXML directory is not the placeholder shown,
set `GT_DIR` to the directory that recursively contains the matching XML files.

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)
