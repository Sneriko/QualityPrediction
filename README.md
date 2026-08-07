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

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)
