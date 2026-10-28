# Hackathon compliance check

Hacktoberfest Open-Source AI Challenge, Week 4: "Ghost in the Machine".

Sources read on 2026-10-29 (JST):

- Challenge page: https://dev.to/challenges/hacktoberfest-week4-2026-10-26
- Contest rules: https://dev.to/page/hacktoberfest-week4-2026-10-26-contest-rules
- Official rules: https://dev.to/page/official-hackathon-rules

Verdicts: **PASS** = evidence checked · **FAIL** = does not comply · **UNVERIFIED** = not yet checked or not yet possible.

## What the rules say (summary)

- Entry period: 2026-10-26 09:00 PDT to 2026-10-31 23:59 PDT (= 2026-11-01 15:59 JST).
- "build a brand-new project with open-source AI at its core". It must be "created during the Entry Period".
- Official rules: "development of your Entry was started during, and not prior to, the Entry Period".
- One entry per person. Entrants must be 18+ (or the age of majority). Japan is not on the excluded list.
- The post must include a demo (deployed link or video), a code link, and an explanation of which open-source AI is used and how the project is built around it.
- Template sections: What I Built / Demo / Code / How I Built It / Why Does Open Innovation Matter?
- Tags: `devchallenge`, `hf26challenge`. Judging weighs Writing Quality most.

## C-01 Entry period

| Item | Verdict | Evidence |
|---|---|---|
| New repository created after 2026-10-26 09:00 PDT | PASS | GitHub `createdAt` 2026-10-28T19:17:14Z (= 2026-10-28 12:17 PDT) |
| Development started inside the period | PASS (self-attested) | The first commit is dated 2026-10-29 JST (`git log --reverse`). The project directory, corpus and code were all created that day. Git dates are set by the author, so this is the author's own attestation. |
| Work completed before deadline | PASS | Code commit c285157 and earlier precede the post |
| Submitted before deadline | PASS | DEV post published 2026-10-28T19:26:47Z (12:26 PDT, deadline 2026-10-31 23:59 PDT): https://dev.to/sakiha6720/i-taught-a-tiny-open-source-language-model-to-speak-halloween-meet-the-haunted-frog-1phf |
| Post-submission commits disclosed in README | PASS | README "Credits and license" notes the documentation-only update made after submission |

## C-02 Novelty

| Item | Verdict | Evidence |
|---|---|---|
| Not a copy of CSE-Frog | PASS | CSE-Frog is used as a pinned PyPI dependency. No engine code is copied (`NOTICE.md`). |
| Stands alone as a Halloween app | PASS | Own package `haunted_frog`, CLI, corpus, terminal |
| Original features | PASS | Corpus and control corpus, train/save metadata, Haunted Terminal, seeded replay, evaluator with 10 conditions, tests |
| Source of existing code credited | PASS | `README.md` "Credits and license", `NOTICE.md` |
| New work visible in commit history | PASS | https://github.com/kagioneko/the-haunted-frog/commits/main |
| No unlicensed reuse | PASS | Only an MIT dependency, used via its API |

Note: CSE-Frog is by the same author and existed before the period (first published to PyPI in early October 2026). The rules require the *entry* to be "brand-new" and started during the period. Our reading is that a pre-existing open-source library used unmodified as a dependency is the open-source AI the entry is built *with*, like any open-weight model, and is not the entry itself. **This is our interpretation, not an organizer ruling.** The DEV post states the timeline plainly. If there is any doubt, ask the organizers before submitting.

## C-03 Open-source AI

| Item | Verdict | Evidence |
|---|---|---|
| CSE-Frog really does the training and generation | PASS | `trainer.py` calls `Frog.learn`. `generator.py` calls `Frog.generate`. `test_output_is_real_model_output` |
| Not just fixed strings | PASS | Frog lines vary by seed (`test_same_seed_same_output` checks that they differ across seeds). Narration is visibly separate. |
| MIT terms respected | PASS | Dependency ships its license. Attribution in `NOTICE.md`. This repo is MIT. |
| Engine/app relationship explained | PASS | `docs/ARCHITECTURE.md` "Where the AI is" |
| Why open-source AI matters | PASS | DEV draft, "Why Does Open Innovation Matter?" |

## C-04 Theme

| Item | Verdict | Evidence |
|---|---|---|
| Ghost in the Machine connection | PASS | A model "living" in the terminal; the corpus includes GHOST IN THE MACHINE, A VOICE IN THE MACHINE, SOMETHING IS IN THE WIRES |
| Halloween world | PASS | Corpus, ASCII frog, orange banner, narration |
| Demo shows it | PASS | `docs/media/haunt.gif` |

## C-05 Deliverables

| Item | Verdict | Evidence |
|---|---|---|
| Code repository URL | PASS | https://github.com/kagioneko/the-haunted-frog (public) |
| Working demo (video or deployment) | PASS | `docs/media/haunt.gif` is embedded in the DEV draft. Anyone can reproduce it by running the repo. |
| Official DEV template used | PASS | The draft uses the template's opening line and section headings, decoded from the challenge page's prefill link. The optional My Agent Session and Prize Categories sections were removed, as the template allows. |
| Post in English | PASS | DEV draft |
| Tags `devchallenge`, `hf26challenge` | PASS | Draft tags: devchallenge, hf26challenge, opensource, python |
| Tech explained | PASS | "How I Built It" section |
| Open innovation explained | PASS | "Why Does Open Innovation Matter?" section |
| Exactly one Week 4 entry | UNVERIFIED | Entrant to confirm |

## C-06 Eligibility and rules

| Item | Verdict | Evidence |
|---|---|---|
| Entrant 18+ | UNVERIFIED | Entrant to confirm |
| Country eligible | UNVERIFIED | Japan is not on the excluded list. The entrant must confirm residence. |
| DEV account in good standing | UNVERIFIED | Entrant to confirm |
| General official rules read | PASS | See the sources above. Key clauses are summarised. |
| No third-party IP violation | PASS | Original corpus and art. MIT dependency. |
| AI tools and assets disclosed | PASS | `README.md` and `NOTICE.md` disclose AI coding assistance. The rules do not require it. |

## Open items before submission

1. ~~Create the public GitHub repo, with a cross-AI review before the push~~ (done: Codex review round 4 PASS-with-nits, nits fixed).
2. ~~Write the DEV post from the official template~~ (done: saved as an unpublished draft). The entrant reviews it and publishes.
3. ~~Publish~~ (done: 2026-10-28T19:26:47Z).
4. Entrant confirms age, residence, DEV account and that this is the only Week 4 entry.
5. ~~Submit before 2026-11-01 15:59 JST and record the URL and time~~ (done, see C-01).
