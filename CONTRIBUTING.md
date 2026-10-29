Contributing
Thanks for looking.  This project is small, and the bar for a change is
simple: it must be measurable, and the measurement must be in the
repository.
The rules
Every claim is a number in `results/`.  If you cannot regenerate it
with `python experiments/run_all.py --full`, it does not go in the docs.
Negative results are first-class.`docs/NEGATIVE-RESULTS.md` is not
an apology section; it is the reason the project can be believed.  If your
change makes something worse, add it there with the numbers.
The self-verification must stay green.`make test` runs 38 checks.
A new feature ships with its check.
Pure NumPy, no new runtime dependencies.  The whole point is that the
thing runs on a laptop, on CPU, with no downloads.  `torch` appears only
in `superfluid/torch_graft.py` and `experiments/gpu_graft.py`, which are
optional and clearly marked.
No silent failures.  If something cannot be done, raise.  The frozen
register refuses to overflow rather than growing; the recall fails loud on
a signature mismatch.  Keep that spirit.
Getting set up
git clone <your-fork>
cd <your-fork>
pip install -r requirements.txt
make test          # the 38 checks
make quick         # a fast smoke run of every experiment
A good first change
Add a check to `selftest.py` for a property you believe in.
Add an experiment under `experiments/` that measures a claim from the
docs, and put the JSON in `results/`.
Fix a bug the selftest found.  There is a `docs/NEGATIVE-RESULTS.md`
section listing the bugs this project found in its own code; the list used
to be longer.
Style
PEP 8, 4 spaces, 88 columns.
Docstrings that say why, not what.
Comments that explain a measurement or a constraint, never the syntax.
Reporting a bug
Open an issue with the smallest reproduction you can manage.  A failing
check in `selftest.py` is the ideal bug report.
Code of conduct
See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).