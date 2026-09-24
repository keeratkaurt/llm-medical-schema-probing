# Autoimmune symptom associations in a language model

An exploratory, contrastive evaluation of whether a pretrained language model assigns lower surprisal to a symptom phrase when it follows its source-associated autoimmune disease name than when the **same phrase** follows another disease name. This is a language-model behavior probe, **not** a diagnostic model.

## Data and design

The 24 hand-selected symptom phrases (six per disease) in [data/symptoms.csv](data/symptoms.csv) are paraphrased from [NIAMS lupus](https://www.niams.nih.gov/health-topics/lupus), [NIAMS rheumatoid arthritis](https://www.niams.nih.gov/health-topics/rheumatoid-arthritis), [NIAMS Sjögren's disease](https://www.niams.nih.gov/health-topics/sjogrens-disease), and [NIDDK celiac disease](https://www.niddk.nih.gov/health-information/digestive-diseases/celiac-disease/symptoms-causes).

For each phrase, the script calculates its mean conditional negative log-likelihood (bits per token) after each of four disease prompts. It ranks diseases by lowest surprisal and reports the source disease's rank and top-1 frequency. Two fixed prompt templates test wording sensitivity. Chance top-1 is 25%.

**Label caution:** These conditions can share symptoms or co-occur. A phrase appearing on one source page does not make its association with the other three diseases clinically false. The label means *source-associated in this curated set*, not medically exclusive. No patient records or clinical notes are used.

## Reproduce

Python 3.10+ on CPU:

```bash
pip install -r requirements.txt
python probe.py --model distilbert/distilgpt2 --output results/distilgpt2
python probe.py --model openai-community/gpt2 --output results/gpt2
```

The model is downloaded from Hugging Face on first run. Each run produces `scores.csv` (192 disease-context scores) and `summary.json` (per-prompt accuracy, breakdown, and bootstrap interval) in its output directory.

## Results

Each of the 24 phrases was scored after each of four disease names under two fixed prompts (192 conditional scores per model). The source disease ranked first in:

| Model | "A patient with ... often experiences" | "A common symptom of ... is" |
| --- | ---: | ---: |
| DistilGPT-2 | 12/24 (50.0%) | 17/24 (70.8%) |
| GPT-2 | 16/24 (66.7%) | 16/24 (66.7%) |

Chance ranking under a uniform four-disease guess is 25%, but the symptom set was hand-selected and not a representative clinical sample. DistilGPT-2 changed by **5 of 24** source matches across prompt templates. Under the first prompt it ranked Sjögren's first for **0 of 6** source phrases; GPT-2 ranked it first for **2 of 6** under both templates. Examples of misranked phrases include dry mouth with trouble swallowing (ranked as lupus by GPT-2) and greasy bulky stools (also ranked as lupus). The model's word probabilities are sensitive to context and may reflect overlapping symptoms, phrasing, and corpus frequency, so these errors cannot be equated with clinical misinformation.

See [DistilGPT-2 summary](results/distilgpt2/summary.json), [GPT-2 summary](results/gpt2/summary.json), and their corresponding `scores.csv` files for each source rank, predicted context, and bits-per-token scores.

## Audit the saved scores

```bash
python analyze.py results/gpt2/scores.csv --output results/gpt2/audit.json
python analyze.py results/distilgpt2/scores.csv --output results/distilgpt2/audit.json
python -m unittest discover -s tests -v
```

The audit verifies that every stored predicted disease and source rank agrees with
the four raw scores, checks that every phrase has both prompts, writes disease-by-
disease confusion counts, and lists misranked phrases with the difference between
the source-associated score and the best score. The checked-in reports show that
the **predicted disease** changes for 6/24 phrases in GPT-2 and 8/24 in
DistilGPT-2. **Whether the source is ranked first** changes for 4/24 and 7/24
respectively; DistilGPT-2 has a net increase of five correct source matches in
the second prompt. This separates the number of changed cases from the net score
change. The reports are descriptive: phrasing and overlapping symptoms remain
uncontrolled confounders. CI checks the analysis against saved outputs without
downloading either model.

## Next experiment

Before claiming robustness, create a larger, separately documented phrase set
with independently sourced descriptions and explicit overlap categories. Freeze
it before running both models, record phrase provenance, include a simple
wording-control baseline, and evaluate uncertainty across source groups. Do not
treat the 24 original phrases as an independent clinical test set or interpret a
source-rank mismatch as a clinical hallucination.

## Limitations

- Tiny, hand-curated, nonindependent symptom set. The bootstrap interval reflects variability within these 24 phrases, not clinical generalization.
- Wording, disease-name familiarity, and pretrained model frequency can drive scores.
- GPT-style token likelihood is not a calibrated probability of a diagnosis.
- The model is a base language model, not a clinical or instruction-tuned model.
- No supervised classifier, clinical hallucination detector, or deployment claim.
