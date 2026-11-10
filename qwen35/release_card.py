"""Shared paper context and citation for PLatour model cards."""
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
