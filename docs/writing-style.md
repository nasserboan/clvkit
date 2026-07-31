# Writing style

Rules for everything written for publication under this project: articles,
README prose, release notes, docs.

The goal is prose that reads like a working engineer wrote it. Most advice on
"making writing sound human" produces writing that sounds like advice on making
writing sound human. The rules below are the ones with teeth.

## The one that matters most

**Be specific.** Every generic sentence reads as machine-written, because
machines produce generic sentences and people producing generic sentences are
not worth reading either.

This project has an unusual surplus of concrete material. Use it.

| Generic | Specific |
|---|---|
| "the library reproduces published estimates" | "α comes out 4.413602 against a published 4.414" |
| "choosing a coarser grain affects the fit" | "α moves 55%, from 4.41 to 6.85" |
| "there was a bug in cohort recovery" | "a Friday acquisition observed to a Wednesday landed a week late" |
| "the incumbent is unmaintained" | "btyd declares `requires_python = '>=3.8,<3.10'`. It will not install on 3.10." |

A number, a date, a file path, a quoted error message. One of those per
paragraph and the prose stops sounding synthetic on its own.

## Banned

**Em dashes at model frequency.** The single loudest tell. Use a comma, a full
stop, or brackets. One em dash per article is a choice; one per paragraph is a
fingerprint. Brackets read as a person interrupting themselves; em dashes at
this density read as a model.

**The rhetorical colon.** Setup, colon, reveal, used as a drumroll. Once is a
choice. Four times in nine hundred words is a fingerprint. Naming a term earns
it ("a name nobody outside it uses: buy till you die"). Building suspense does
not. Keep the colons introducing a formula, a list or a quoted value, and turn
the rest into full stops.

**"Not just X, but Y."** And its family: "It's not about X. It's about Y."
"More than just X." State the thing. If Y is the point, write Y.

**The rule of three.** Models list three items, always. Real lists have two,
four, or one. If a list has exactly three items, check whether the third was
invented to fill the pattern, and cut it if so.

**Hedges.** "may", "might", "could potentially", "generally speaking", "it is
important to note", "arguably". Somebody who knows the answer states it. If the
claim is uncertain, say what would settle it.

**Weightless adjectives.** "powerful", "robust", "seamless", "comprehensive",
"leverage", "delve", "landscape", "tapestry", "testament", "realm", "crucial",
"vital", "quiet" (quiet confidence, quiet rebellion, quietly growing).

**Transitions that announce themselves.** "Furthermore", "Moreover", "In
addition", "It's worth noting that", "That said" as a tic. Paragraphs that
follow each other logically do not need a signpost saying so.

**Uniform rhythm.** Sentences all the same length is the structural tell.
Vary them. A short one lands.

**Copula avoidance.** "serves as", "stands as", "represents", "boasts",
"functions as", all standing in for "is" and "has". One is ordinary English.
Four in a row is a model refusing to repeat a verb. "The BG/NBD serves as the
frequentist baseline" is "The BG/NBD is the frequentist baseline".

**Participles that fake depth.** A clause bolted to the end of a sentence to
imply analysis: "showcasing", "highlighting", "underscoring", "reflecting",
"emphasizing". "α comes out 4.413602, highlighting the library's accuracy" says
nothing the number hasn't already said. Delete the clause, or promote it to its
own sentence with a source in it.

**Synonym cycling.** Rotating "customer", "client", "buyer" and "purchaser"
through one paragraph to avoid repeating a word. Models do this because a
repetition penalty tells them to. Pick the term the domain uses and repeat it.
Here that word is "customer", every time.

**False ranges.** "From X to Y" where X and Y aren't the ends of any scale.
"Everything from data cleaning to executive dashboards" is two items pretending
to be a spectrum. Write the list.

**Boldface as emphasis.** Bold marks a term on first mention. It doesn't mark
the sentence you want read hardest. Three bolded phrases in a nine hundred word
article is a model reaching for a highlighter, and the reader stops trusting any
of them.

**The tidy closer.** "In conclusion", "Ultimately", "At the end of the day",
and any final paragraph that restates the piece. Stop when the argument stops.

**Metaphors that do not survive inspection.** If the comparison breaks under
one question, cut it. A tuning-a-guitar analogy for a business process is worse
than no analogy.

**Marketing tone.** No "revolutionary", no "game-changing", no exclamation
marks. The library is a set of likelihood functions. It is allowed to be one.

## Required

**Admit things.** The most human sentence available is one that concedes a
defect. This project has real ones on record: a spec that promised something
the code could not do, a display module that violated the two rules stated in
its own docstring, a CI action version that never existed, tests that passed
without testing anything. Writing that includes them cannot be mistaken for
marketing copy.

**Name the cost.** Every design choice here has a price and the repo already
states most of them in `opinions.md`. A survival curve is a cross-section
across cohorts, not a panel followed forward. The retention matrix is
float-typed even though it counts customers. Saying so is what a practitioner
does.

**Show the number that surprised you.** Not the number that confirms the
argument.

**Quote sources exactly.** Section numbers, equation numbers, verbatim
sentences. "Fader, Hardie & Lee, §2.2" beats "the literature suggests".
Precision is the cheapest credibility available and models generate vague
attribution by default.

**Write to one reader.** For this project's articles that reader is an analyst
who believes CLV is `average ticket × frequency × margin`, and who has never
heard of `lifetimes`. Address them, not a search engine.

**Keep contractions.** "doesn't", "isn't", "won't". Prose without contractions
reads formal in the specific way that generated text reads formal.

## Before publishing

Read it aloud. The parts that are hard to say out loud are the generated parts.

Then check:

- Em dash count. If it exceeds one or two in the whole piece, rewrite them out.
- Colon count, minus the ones introducing a formula, a list or a quoted value.
  What is left is the rhetorical ones, and more than one or two is a pattern.
- Any paragraph that could appear in an article about a different library. Cut
  or make it specific.
- Any claim a reader could falsify in thirty seconds. Verify it or drop it.
  A false claim about a competitor destroys the credibility of every true claim
  beside it.
- The last paragraph. If it summarises, delete it.

**Count once.** When several of these land on the same phrase, boldface plus a
coined term plus a dramatic aside, that is one strong tell and not three.
Counting it three times inflates the problem and sends you rewriting a sentence
that needed a single fix.

## Provenance

The bans on copula avoidance, participles that fake depth, synonym cycling,
false ranges and boldface were collected from published guidance on spotting
synthetic prose. The rhetorical colon is ours.
