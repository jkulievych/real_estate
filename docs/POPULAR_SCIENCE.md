# Can an Algorithm Tell You If You're Overpaying for an Apartment?

*A data science experiment with 7,116 Romanian real estate listings*

---

Imagine you're apartment hunting in Cluj-Napoca. You find a 52 m² flat on the fourth floor, two rooms, built in 1985, no parking. The seller is asking €89,000. Is that a fair price? Too high? A steal?

Unless you spend weeks trawling listings and building a mental model of the market, there's no easy answer. Real estate pricing is famously opaque — the same apartment in the same city can sell for wildly different prices depending on the street, the floor, the negotiating skill of the parties involved, and frankly, the mood of the market on a given Tuesday.

This project is an attempt to make that opacity a little more transparent, using nothing but publicly available data and some mathematics.

---

## Where the Data Comes From

The starting point is [imobiliare.ro](https://imobiliare.ro), Romania's largest property listing portal. Every listing page contains a hidden block of structured data — a machine-readable summary that the website publishes for search engines. By collecting those blocks for 7,116 listings across Romanian cities, we end up with a reasonably clean dataset: price, area, number of rooms, floor, year of construction, city, and a handful of amenities like parking or balcony.

This is what data scientists call "scraping", and it's less dramatic than it sounds — essentially, automated reading of publicly available web pages.

---

## Teaching a Computer What Affects Price

Raw data isn't a model. The interesting step is figuring out which factors actually predict price, and how strongly.

Some things are obvious: larger apartments cost more. More rooms cost more. Being in Bucharest costs more than being in a small town. But the relationships are rarely simple. A 60 m² flat on the 10th floor of a 1970s panel block behaves very differently from a 60 m² flat in a newly built boutique building, even in the same neighbourhood.

To capture these nuances, we trained two models on the dataset:

**The first model — Ordinary Least Squares (OLS)** — is the classic, interpretable approach. It finds the straight-line relationship between each feature and price, expressed as a simple formula. If you've heard of linear regression, this is it. The advantage: you can read off the coefficients. "Each additional square metre is worth approximately €X. Being on the ground floor costs approximately €Y relative to higher floors." It's a model you can explain to a non-statistician.

**The second model — Random Forest** — is more complex. It builds hundreds of decision trees, each trained on a random subset of the data, and averages their predictions. It handles non-linear relationships and interactions between features much better than OLS. In practice, it's more accurate — but also more of a black box.

Both models are trained, evaluated, and saved. The app uses Random Forest by default because its predictions are measurably better; OLS is included for the sake of interpretability and transparency.

---

## What "Confidence Interval" Actually Means

When the app estimates a price, it doesn't just give you a single number. It gives you a range — something like **€82,000 – €96,000** around the central estimate of €89,000.

This range is called a confidence interval, and it's there to prevent false precision. A model trained on 7,000 listings cannot know everything about every apartment. There's always uncertainty. The interval communicates that uncertainty honestly: *given what the model knows, it's reasonably confident the true market value falls within this range*.

The intervals are computed using a technique called **bootstrap resampling** — the model is run many times on slightly shuffled versions of the data, and the spread of those predictions defines the interval. No single apartment is exactly like another; the confidence interval acknowledges that.

---

## The App

All of this is wrapped in a desktop application. You enter the apartment's details — area, rooms, floor, city, layout type — click a button, and the model returns a price estimate with its confidence band and a list of comparable real listings from the dataset.

The app also keeps a local history of every estimate you've run, stored in a small database on your computer. No data leaves your machine.

It's not a professional valuation tool. It's a starting point — a way to gut-check a listing before you commit to a viewing, or to understand whether the asking price is in the neighbourhood of what the market expects.

---

## What the Data Reveals

A few things that emerged from exploring the dataset:

- **Area matters most**, as expected — it accounts for the largest share of price variance across all cities.
- **City matters enormously.** Bucharest and Cluj-Napoca command prices 40–60% higher than cities like Craiova or Galați for comparable apartments.
- **Layout type is surprisingly important.** A *decomandat* layout (rooms separated by a hallway, not connected to each other) consistently commands a premium over *semidecomandat*, even controlling for area and room count.
- **Floor has a non-linear effect.** Ground floors are discounted. Top floors without a lift are also discounted. The "sweet spot" is somewhere in the middle — but this varies by building type.
- **Year of construction is a noisy signal.** Older communist-era blocks and new-builds are both present in abundance; the relationship between age and price depends heavily on whether the building has been renovated.

None of these findings would surprise an experienced real estate agent in Bucharest. The interesting part is that a machine learning model can learn these patterns from raw data, without being told any of them explicitly.

---

## Limitations

This project is honest about what it cannot do:

- **No hyperlocal data.** The model knows the city, not the street. Two apartments 200 metres apart can have very different values based on proximity to a metro station, a park, or a noisy road — and the model cannot see that.
- **No condition data.** A freshly renovated flat and an unrenovated one in the same building look identical to the model unless explicitly flagged in the listing.
- **Snapshot in time.** The data was scraped in May 2025. Real estate markets move. A model trained on historical data will gradually drift from current prices.
- **Listings ≠ transactions.** The dataset contains asking prices, not sale prices. Actual deals happen at a discount (or occasionally a premium) from the listed price, and that gap varies by city, season, and market conditions.

These aren't failures of the model — they're inherent limits of the data. A good data scientist reports them.

---

## The Bigger Picture

What this project demonstrates is not that machines are better at pricing apartments than people. They aren't — not yet, and perhaps not for a long time.

What it demonstrates is that **structured thinking about data can make markets more legible**. When you can look at a number and say "the model expects something in this range, and this listing is 30% above that — I should ask why", you're in a stronger position as a buyer or seller.

Data doesn't eliminate uncertainty in real estate. But it can change uncertainty from vague unease into something more tractable: a number with a range, a prediction with caveats, a starting point for a more informed conversation.

---

*Built as a coursework project using Python, scikit-learn, and Flet. Data sourced from imobiliare.ro for educational purposes.*
