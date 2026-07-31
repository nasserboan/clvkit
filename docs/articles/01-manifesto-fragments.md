# I thought CLV was a multiplication

I did not know CLV could be that deep.

---

The libraries were old and out of date. That is the part everyone tells you.
The part I do not hear said out loud is the other reason: I wanted my own
library, for the analyses I kept doing by hand.

---

I was a data scientist doing this work. I am a machine learning engineer now.
The analyses did not stop being needed when I changed titles.

---

The reader I am writing for believes this:

    CLV = average ticket × frequency × margin

They are not stupid and the formula is not a strawman. It is the first result
you get when you search for how to calculate customer lifetime value. It gives
a number. The number goes in a slide.

---

And then nothing happens. They cannot say what the number would have to be for
the decision to flip. They cannot defend it when someone asks where it came
from. So the number goes in the slide and the slide goes in a folder.

---

Two things I believe that not everyone does:

Without auditability, none of it matters. A lifetime value nobody can trace is
a lifetime value nobody will act on.

And frequency is almost always computed wrong.

---

**The gap.** The frequency in that formula and the frequency the model wants
differ by exactly one. The one is the first purchase.

The naive formula counts it. BTYD does not, and for a reason that survives
being said out loud: a first purchase proves nothing about recurrence. It only
proves the person showed up.

---

A customer with five purchases has `frequency = 4`. A customer with one
purchase has `frequency = 0`, and `recency = 0`, and `monetary_value = 0`. Not
missing. Zero. That customer is not evidence of anything yet.

On CDNOW, 1,411 of 2,357 customers are that customer. Sixty percent.

---

Spiritual successor. Grandfather and grandmother, both gone.

(Weaker than it sounds: `lifetimes` and `btyd` are one line, not a couple —
btyd is a literal git fork of lifetimes. And I share no code with either.
What I inherited was the conventions, not the implementation.)

---

`lifetimes` and `btyd` are not my ancestors. They are my siblings — other
implementations of the same papers. The ancestor is Fader & Hardie. I did not
inherit from `lifetimes`; I inherited from the place `lifetimes` inherited
from.

Which is why forking was never the answer. The source code that mattered was
the paper.

---

`btyd` did not merely stop. It declares `requires_python = ">=3.8,<3.10"`. It
will not install on Python 3.10. It never left beta; the last release was
0.1b3, November 2022.

---

Both of their READMEs now point at the same replacement. Two dead frequentist
libraries, sending their readers to a Bayesian one.

---

R still has this. `CLVTools` is on CRAN, not archived, with Pareto/NBD, BG/NBD,
Gamma/Gompertz/NBD and covariates. So the hole is not in the field. The hole is
in Python.

---

The thing that hooked me: this is not a heuristic dressed up. It is a
likelihood. Buying happens as a Poisson process with rate λ; λ varies across
people as a gamma; after any purchase a customer may quietly stop, with
probability p; p varies as a beta. Four parameters and the whole population
falls out.

Nobody ever tells you they left. That is the entire problem, and the model is
built around the fact that the event is never observed.

---

What "deep" turned out to mean, concretely: a Gaussian hypergeometric function
shows up in the conditional expectation, and it is not decoration. The integral
over the beta-distributed dropout probability *is* Euler's integral
representation of ₂F₁. The special function is what the assumption becomes when
you do the algebra.

---

The proof it is right: `α` comes out 4.413602 against 4.414 published. `r`
0.242595 against 0.243. Not transcribed. Fitted, in a test, on every pull
request.

---

The paper has a screenshot of an Excel worksheet with four cells in it. That is
the target. Twenty years later the target still holds.

---

**The shrug.**

I had just built the standard CLV. The wrong one. I presented it to the CMO of
the company I was working at, and he shrugged. Not hostile. He simply
understood that the thing I had taken a long time to build was not important.

---

I went home and asked myself two questions. Was I doing the analysis wrong, or
was I failing to translate it into a business problem?

It was both.

---

That is the whole reason this library exists, and it is also why the formula is
not the only villain. A correct number that nobody can act on gets the same
shrug as a wrong one.

---

Working out what the CMO actually needed took longer than working out what was
wrong with the arithmetic.

---

The shrug is the thing to design against. Not "is this statistically sound",
but "would this survive contact with someone who has a budget to move".

---

There is a version of this story where I blame the CMO. He was right. I had
spent a week producing a number he could not do anything with.

---

What he actually wanted, in his words: **who is worth retaining, and how much
to spend on it.**

Two questions. I had answered neither. I had answered a third one nobody asked:
what did these people spend last year.

---

The formula cannot answer either of them, and it is worth being precise about
why.

*Who is worth retaining* needs two things the formula has no room for: some
notion of who is at risk of leaving, and some notion of who is worth keeping.
It has neither. It ranks by past value, and past value is a bad proxy for
future risk. Your highest-spending customer might have quietly stopped six
months ago and the formula will still rank them first.

*How much to spend* needs a horizon and a discount rate. Spending R$100 to
retain someone worth R$500 over ten years is a different decision from R$500
over one year, and the formula produces the same number for both.

---

Retention money spent on a customer who was never going to leave is wasted, and
it is invisible waste. The campaign reports a conversion. The customer would
have bought anyway.

---

The formula collapses two questions into one number. Will they come back, and
how much will they spend when they do. Those separate cleanly, and once
separated you can rank on either.

---

The CMO asked a decision question and I handed him a descriptive number. He was
not shrugging at the statistics. He was shrugging at the category error.

---

**How it actually ended.**

He went and looked at Google Analytics charts, and made the decision he already
wanted to make.

---

That is the part I think about most. My analysis did not lose because it was
wrong. It lost to a chart that was already open, in a tool he already trusted,
which happened to agree with him.

Rigour was never what it was competing on.

---

So the lesson is not "be more correct". I was going to be more correct anyway.
The lesson is that a number nobody can interrogate loses to a chart somebody
can, every time, and it deserves to.

---

This is why I care about auditability more than I care about the model being
sophisticated. Not as a principle. Because I watched the alternative lose in a
meeting.

---

Google Analytics did not beat me on statistics. It beat me on being there,
being legible, and never asking anyone to take it on faith.

---

If the number cannot survive the question "where did this come from", it does
not matter how good the likelihood was.

---

There isn't one naive formula. Search "how to calculate CLV" and you get
several, and they disagree with each other. Some multiply by a retention rate,
some divide by churn, some add a discount factor, some drop margin entirely.

The plurality is the tell. If the question had a settled answer, there would be
one formula. What you're actually looking at is a family of guesses, each
plausible, none anchored to anything you could test.

**VERIFIED, 25 July 2026.** Query: "how to calculate customer lifetime value
formula". Top three results, five formulas between them:

1. NetSuite — `CLV = Average Transaction Size x Number of Transactions x
   Retention Period`. No margin, no discount rate. Worked example:
   `$4 x 100 x 5 = $2,000`.
2. HubSpot — `Customer Lifetime Value = Customer Value x Average Customer
   Lifespan`, where `Customer Value = Average Purchase Value x Average Number
   of Purchases`. No margin, no retention rate, no discount rate.
3. Qualtrics — three formulas on one page: `(customer value * average customer
   lifespan)`; `Customer revenue per year * Duration of the relationship in
   years - Total costs of acquiring and serving the customer`; and
   `GML * Retention rate / (1+ Rate of discount - Retention rate)`.

The number that surprised me: neither of the top two results includes margin,
so the formula this article attributes to the reader is itself only one member
of the family. Said in the article rather than glossed over.

Sources: netsuite.com/portal/resource/articles/ecommerce/customer-lifetime-value-clv.shtml
(the .com page returns 403 to fetchers; text confirmed on the .com.sg mirror),
blog.hubspot.com/service/how-to-calculate-customer-lifetime-value,
qualtrics.com/experience-management/customer/how-to-calculate-customer-lifetime-value/
