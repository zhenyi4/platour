"""Documentation templates for PLatour model cards; no model execution or export."""
import re

PAPER_URL = 'https://arxiv.org/html/2611.06976'
PAPER_TITLE = 'On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer'
PAPER_SECTION = f'''## Paper

[{PAPER_TITLE}]({PAPER_URL})

PLatour explores Prefill-to-Autoregressive Cross-model Transfer (PACT): a larger sender encodes the prompt and supplies hidden representations, while a smaller receiver generates the response independently. The main method trains lightweight linear bridges between frozen backbones using task rewards, starting from zero-initialized projections and supporting hybrid-attention models.

The paper also introduces PACT-Check, with Dummy Transfer, Irrelevant Transfer, Self-Communication and Trained Receiver controls. These comparisons examine whether improvements depend on useful sender information or can arise through adaptation of the receiver. Better task performance alone does not demonstrate effective cross-model communication.
'''

BIBTEX = '''@misc{pact_platour,
  title={On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer},
  author={Zhenyi Shen and Junru Lu and Wei Liu and Jiangnan Ye and Jiazheng Li and Di Yin and Xing Sun and Tameem Adel and Yali Du and Lin Gui and Yulan He},
  year={2026},
  eprint={2611.06976},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2611.06976},
}'''
CITATION_SECTION = '## Citation\n\n```bibtex\n' + BIBTEX + '\n```\n'

PROJECT_URL = 'https://github.com/zhenyi4/platour'
BASE_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def project_setup():
    return f'''## Project environment

Use Linux with CUDA, Python 3.11 and enough GPU memory for the selected models.
Run the project commands from an editable checkout so the release manifests and
task files are available. See the [installation guide]({PROJECT_URL}#getting-started).

```bash
git clone https://github.com/zhenyi4/platour.git
cd platour
python3.11 -m pip install --user uv
python3.11 scripts/install_environment.py
source .venv/bin/activate
```

Training, generation and scoring use the active Python. The examples below need
no `vllm_python` argument. See the [environment guide]({PROJECT_URL}/blob/main/docs/ENVIRONMENT.md)
for installation options, runtime checks and compatibility details.
'''


def project_generation(record):
    identifier = record['id']
    virtual = record['artifact_type'] == 'virtual_sender'
    self_communication = identifier in (21, 27)
    method = ('virtual_sender' if virtual else 'self_communication' if self_communication
              else record['method'])
    lines = ['from platour_exps import ensure_weight, resolve_model_input',
             'from platour_api import load_session', '',
             '# Download/cache backbones; verify the released checkpoint against the manifest.',
             f'receiver = resolve_model_input("{record["receiver"]}")']
    if record['sender'] and not self_communication:
        lines.append(f'sender = resolve_model_input("{record["sender"]}")')
    lines += [f'checkpoint = ensure_weight({identifier})', '', 'system = load_session(',
              f'    family="{record["family"]}", method="{method}",',
              '    receiver=receiver, bridge=checkpoint,']
    if record['sender'] and not self_communication:
        lines.append('    sender=sender,')
    if record['family'] == 'qwen3':
        lines.append(f'    mapping="{record["layer_mapping"]}",')
    else:
        lines.append(f'    task="{record["task"]}",')
    lines.append(')')
    question = ('Write a Python function that returns the sum of two integers.'
                if record['task'] == 'deepcoder' else 'What is 2 + 2?')
    if virtual:
        prefix, suffix = (42, 22) if record['task'] == 'dapo' else (45, 23)
        suite = 'aime' if record['task'] == 'dapo' else 'lcb'
        lines += ['', 'import json', 'from pathlib import Path',
                  'from platour_data import ensure_qwen35_data', '',
                  '# Keep the released task template; replace only its question span.',
                  'data_root = Path("data/qwen35")',
                  f'ensure_qwen35_data({identifier}, data_root, "evaluate")',
                  f'with (data_root / "{record["task"]}/{suite}.jsonl").open() as stream:',
                  '    example = json.loads(next(line for line in stream if line.strip()))',
                  'template = example["prompt_token_ids"]',
                  f'question = "{question}"',
                  f'prompt_ids = (template[:{prefix}]',
                  '              + system.tokenizer.encode(question, add_special_tokens=False)',
                  f'              + template[-{suffix}:])']
    lines += ['', 'answer = system.generate(',
              ('    prompt_ids=prompt_ids,' if virtual else
               f'    messages=[{{"role": "user", "content": "{question}"}}],'),
              '    backend="vllm", max_new_tokens=256,',
              f'    output="./outputs/{identifier:02d}-custom-prompt",',
              ')', 'print(answer["response"])']
    description = ('''This checkpoint contains bridge projections and learned virtual template features.
It needs the receiver only. Its fixed template is required: arbitrary chat
messages are not supported. The example below reuses the released template and
replaces its question span without modifying any dataset file.''' if virtual else
                   '''Self-Communication uses the same public 4B backbone as sender and receiver,
plus this trained bridge; no larger external sender is needed.''' if self_communication else
                   '''This checkpoint contains bridge projections, not complete model weights.
Load the separately sourced frozen sender and receiver with the project API;
`AutoModel.from_pretrained` cannot load the bridge as a standalone model.''')
    body = ('## Generate an answer\n\n' + description + '\n\n'
            'After the project setup above, run this Python example from the checkout root.\n'
            'The API receives local backbone directories and a local checkpoint file;\n'
            '`resolve_model_input` and `ensure_weight` prepare those paths first.\n\n'
            '```python\n' + '\n'.join(lines) + '\n```\n\n'
            'This is a custom-prompt demo, not the published benchmark protocol. The 256-token\n'
            'limit is only a short example; increase it for longer answers. Use a fresh output\n'
            'directory for each run. Generation returns text and does not execute generated code.\n')
    if record['family'] == 'qwen3':
        body += ('\nQwen3 PLatour-SFT uses `last_aligned`; PLatour (GRPO) uses\n'
                 '`uniform_endpoints`. The example sets this checkpoint\'s mapping explicitly.\n')
    return body


def standard_generation(record):
    lora = record['method'] == 'lora'
    repo = 'zen-E/' + record['slug']
    lines = ['import torch']
    if lora:
        lines += ['from peft import PeftModel',
                  'from transformers import AutoModelForImageTextToText, AutoTokenizer', '',
                  'base_id = "Qwen/Qwen3.5-4B"', f'revision = "{BASE_REVISION}"',
                  'base = AutoModelForImageTextToText.from_pretrained(',
                  '    base_id, revision=revision, dtype="auto", device_map="auto",',
                  ')', f'model = PeftModel.from_pretrained(base, "{repo}")',
                  'tokenizer = AutoTokenizer.from_pretrained(base_id, revision=revision)']
    else:
        lines += ['from transformers import AutoModelForCausalLM, AutoTokenizer', '',
                  f'repo = "{repo}"',
                  'model = AutoModelForCausalLM.from_pretrained(',
                  '    repo, dtype="auto", device_map="auto",', ')',
                  'tokenizer = AutoTokenizer.from_pretrained(repo)']
    question = ('Write a Python function that returns the sum of two integers.'
                if record['task'] == 'deepcoder' else 'What is 2 + 2?')
    lines += ['model.eval()', '',
              f'messages = [{{"role": "user", "content": "{question}"}}]',
              'inputs = tokenizer.apply_chat_template(',
              '    messages, tokenize=True, add_generation_prompt=True,',
              '    return_dict=True, return_tensors="pt",',
              ').to(model.get_input_embeddings().weight.device)',
              'with torch.inference_mode():',
              '    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)',
              'new_tokens = output[0, inputs["input_ids"].shape[-1]:]',
              'print(tokenizer.decode(new_tokens, skip_special_tokens=True))']
    note = ('Use `AutoModelForImageTextToText` for the official multimodal backbone, then\n'
            'attach the PEFT adapter. Only the text decoder is adapted. The adapter does\n'
            'not include backbone weights; all original FP32 adapter values and fused QKV\n'
            'modules are preserved. This differs from the project\'s vLLM runtime conversion.\n'
            if lora else
            'This is a complete text receiver, including its configuration and tokenizer;\n'
            'no external sender or separate receiver download is needed for this example.\n'
            'There is no vision encoder. The exported input embeddings and output head\n'
            'remain independent, preserving the original BF16 tensors.\n')
    return ('## Generate an answer with Transformers\n\n' + note + '\n'
            'Activate the project environment from the setup section above.\n'
            'It includes Transformers and PEFT for this standard loading route.\n\n'
            '```python\n' + '\n'.join(lines) + '\n```\n\n'
            'This short custom-prompt example does not reproduce the benchmark protocol.\n'
            'Increase `max_new_tokens` for longer answers. Generating code requires no coding\n'
            'judge; running benchmark tests does.\n')


def benchmark_sections(record):
    identifier = record['id']
    repo = 'zen-E/' + record['slug']
    family = 'qwen3' if record['family'] == 'qwen3' else 'qwen35'
    command = [f'platour-exps evaluate --experiment {identifier}', f'  --weights {repo}']
    if record['sender']:
        command.append(f'  --sender {record["sender"]}')
    command += [f'  --receiver {record["receiver"]}']
    if record['task'] == 'deepcoder':
        command += ['  --code-cases "$JUDGE_ROOT/cases"',
                    '  --official-checker "$LCB_CHECKER"',
                    '  --docker-image "$JUDGE_IMAGE"']
    command.append(f'  --output ./outputs/{identifier:02d}-release-eval')
    before = ''
    if record['task'] == 'deepcoder':
        before = (f'First complete the [coding judge setup]({PROJECT_URL}/blob/main/qwen35/README.md#obtain-and-install-the-coding-judge-ids-2530),\n'
                  'which defines `JUDGE_ROOT`, `LCB_CHECKER`, `JUDGE_IMAGE` and `PYTHONPATH`.\n'
                  'The command below requires those values, the prepared cases, and Docker.\n'
                  'Generated programs are tested inside isolated containers.\n\n')
    body = f'''## Evaluate the released checkpoint

Run from the project environment above. This uses the published TEST inputs and
the selected experiment's sampling/scoring settings; no training is performed.

{before}```bash
''' + ' \\\n'.join(command) + f'''
```

`--sender` and `--receiver` accept HF IDs or local directories. Missing backbones,
released weights and published task inputs download automatically. The release
loader verifies the configuration and weight hashes. Omitting `--weights` selects
the same experiment's published checkpoint. Use a new output directory.

To download and verify just the released checkpoint without loading models:

```bash
platour-exps download --experiment {identifier}
```

See the [task protocols]({PROJECT_URL}/blob/main/{family}/README.md) and
[checkpoint index]({PROJECT_URL}/blob/main/models/README.md#downloadable-checkpoints).
'''
    if record['task'] == 'dapo':
        body += ('\nThe default evaluates AIME, HMMT and IMO with the published 32K response cap.\n'
                 'Add `--eval-suites aime` for AIME-135 and its Clean80 subset only; partial\n'
                 'runs report `three_suite_macro: null`. Adding `--response-cap 16384` shortens\n'
                 'the cap and is recorded as a change from the published test protocol.\n')
    elif record['task'] == 'deepcoder':
        body += ('\nThe default scores LCB v6, HumanEval and MBPP. The judge is required for\n'
                 'benchmark scoring, not for the custom-prompt generation example.\n')
        if identifier == 28:
            body += ('Virtual Sender uses the original LCB prompt and explicitly template-adapted\n'
                     'HumanEval/MBPP prompts; retain that distinction when reporting results.\n')
    group = 2 if identifier in (*range(1, 7), 13, 14, 15) else 4 if identifier in (20, 25) else 1
    body += f'''
## GPU placement for project evaluation

| Option | Behavior for this checkpoint |
|---|---|
| Omit `--n-gpu` | One process with the existing automatic placement; no automatic data-parallel replicas. |
| `--n-gpu 8` | {8 // group} independent replicas, each using {group} GPU{'s' if group > 1 else ''}, with disjoint question shards. |
| `--n-gpu N --gpus-per-replica G` | N total GPUs split into N / G replicas; N must be divisible by G. |

The default group size for this checkpoint is **{group} GPU{'s' if group > 1 else ''} per replica**.
'''
    body += ('`--n-gpu 1` is supported when the complete workload fits on one GPU.\n' if group == 1 else
             '`--n-gpu 1` is incompatible with this default group size. Group sizes are\n'
             'placement settings, not a guarantee that the model fits in memory.\n')
    body += (f'Every replica needs enough memory for its models and decoding workload.\n'
             'Use `CUDA_VISIBLE_DEVICES` to select/order GPUs. Do not combine `--n-gpu`\n'
             'with `torchrun`. These flags apply to the project evaluator, not the Python\n'
             f'`generate()` examples. See [evaluation GPU defaults]({PROJECT_URL}#gpu-defaults-for-standalone-evaluation).\n')
    return body


def usage_sections(record, *, standard_format=False):
    links = (f'[Code]({PROJECT_URL}) · '
             f'[Model collection](https://huggingface.co/collections/zen-E/platour) · '
             f'[Agent/API guide]({PROJECT_URL}/blob/main/AGENT.md)\n\n')
    if standard_format:
        return links + project_setup() + '\n' + standard_generation(record) + '\n' + benchmark_sections(record)
    if record['method'] in ('full', 'lora'):
        raise ValueError('Use the standard published Full/LoRA card, not legacy project packaging')
    return links + project_setup() + '\n' + project_generation(record) + '\n' + benchmark_sections(record)


def update_usage_sections(card, record):
    """Replace usage prose only, preserving metadata, provenance and citation text."""
    standard = record['id'] in (23, 24, 29, 30)
    links = f'[Code]({PROJECT_URL})'
    if links in card:
        # Published cards already have expanded usage sections. Replace exactly
        # that block; keep front matter, paper, provenance and citations intact.
        end = ('## Export dependency and scope\n' if standard else
               '## Evaluation provenance\n')
        if card.count(links) != 1 or card.count(end) != 1:
            raise ValueError('Expected one published usage block and provenance section')
        start, stop = card.index(links), card.index(end)
        if stop <= start:
            raise ValueError('Published provenance must follow the usage block')
        return (card[:start] + usage_sections(record, standard_format=standard).rstrip()
                + '\n\n' + card[stop:])
    start = '\n## Load\n' if standard else '\n## Loading\n'
    end = '\n## Export dependency and scope\n' if standard else '\n## Reproducibility and licensing\n'
    if card.count(start) != 1 or card.count(end) != 1:
        raise ValueError('Expected one original usage section and its following provenance section')
    head, rest = card.split(start, 1)
    _, tail = rest.split(end, 1)
    provenance = ''
    if not standard and '\n## Evaluation provenance\n' in head:
        head, notice = head.split('\n## Evaluation provenance\n', 1)
        provenance = '\n## Evaluation provenance\n' + notice.rstrip() + '\n'
    return (head.rstrip() + '\n\n' + usage_sections(record, standard_format=standard).rstrip()
            + '\n' + provenance + end + tail)


def dataset_usage(group):
    identifier = {'qwen3': 11, 'dapo': 19, 'deepcoder': 26}[group]
    family = 'qwen3' if group == 'qwen3' else 'qwen35'
    body = f'''## Use with PLatour

For the project training/evaluation CLI, follow the
[installation guide]({PROJECT_URL}#getting-started) and activate the project
environment. From the checkout root, download and verify one experiment's inputs:

```sh
platour-exps download-data --experiment {identifier}
```

Training and evaluation also download missing signed inputs automatically.
Generation uses the active Python; no separate decoder path is required.
See the [task recipes]({PROJECT_URL}/blob/main/{family}/README.md) and
[data guide]({PROJECT_URL}/blob/main/data/README.md) for all experiments,
split contracts and offline use.
'''
    if group == 'deepcoder':
        body += (f'\nProgramming evaluation also requires the '
                 f'[coding judge setup]({PROJECT_URL}/blob/main/qwen35/README.md#obtain-and-install-the-coding-judge-ids-2530).\n')
    return body


def update_dataset_usage(card, group):
    """Add or refresh CLI instructions without regenerating dataset metadata."""
    start, end = '## Use with PLatour\n', '## Sources and use\n'
    if card.count(end) != 1 or card.count(start) > 1:
        raise ValueError('Expected one dataset source section and at most one usage section')
    stop = card.index(end)
    begin = card.index(start) if start in card else stop
    if begin > stop:
        raise ValueError('Dataset usage must precede source notices')
    return card[:begin] + dataset_usage(group).rstrip() + '\n\n' + card[stop:]


def add_paper_sections(card):
    """Preserve the card body; add paper context near its title and citation last."""
    if '\n## Paper\n' not in card:
        title = re.search(r'^# [^\n]+\n', card, flags=re.M)
        if title is None:
            raise ValueError('Model card needs a title before adding paper context')
        card = card[:title.end()] + '\n' + PAPER_SECTION + '\n' + card[title.end():].lstrip('\n')
    if '\n## Citation\n' in card:
        if not card.rstrip().endswith(CITATION_SECTION.rstrip()):
            raise ValueError('Existing citation differs; review it before replacing')
    else:
        card = card.rstrip() + '\n\n' + CITATION_SECTION
    if card.startswith('---\n'):
        end = card.find('\n---\n', 4)
        if end < 0:
            raise ValueError('Unclosed model card metadata')
        metadata, body = card[:end], card[end:]
        tag = '  - arxiv:2611.06976'
        if tag not in metadata.splitlines():
            if '\ntags:\n' in metadata:
                metadata = metadata.replace('\ntags:\n', '\ntags:\n' + tag + '\n', 1)
            else:
                metadata += '\ntags:\n' + tag
        card = metadata + body
    return card
