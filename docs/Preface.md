---
hide:
  - toc
---

### Why This Book Exists

This book wasn’t born from mastery.
It was born from **frustration**, and a deep desire to understand.

I come from the world of **firmware engineering**, a world where precision matters, memory is tight, and every line of C++ can be traced down to a register or a flag. When I started exploring machine learning, I expected the same clarity.

But instead, I found abstraction.

Libraries like `scikit-learn` offered powerful APIs, `.fit()`, `.predict()`, pipelines, cross-validation, hyperparameter tuning, but I often felt like I was **pushing buttons on a machine I didn’t understand.**

What exactly *is* an SVM doing under the hood?  
What’s a “kernel”?  
What does `GridSearchCV` *really* do behind the scenes?  
How does `StandardScaler` impact model performance—and why?  

And more importantly:

> How can I trust these tools, if I don’t know how they work?

I wanted more than toy examples. I wanted to **trace the math**, understand the **design of the library**, and see **how classical ML algorithms actually operate**, both as mathematical models and as real Python objects.

But most ML books either:

* Dived straight into the math, assuming a strong background in statistics…
* Or stayed at the surface level, teaching only “what button to press” in a notebook.

There was little for humble tinkerers, far from experts, who **think like engineers**, question everything, and want to **go deep before going fast**.

So I started collecting notes, drawing diagrams, tracing source code, and building mental models.

Those notes became a system. That system became a narrative.
And now, it’s this book.

---

### Who Should Read This

This book is for:

* **Engineers from systems or embedded backgrounds** who are transitioning into machine learning but struggle with abstraction-heavy tutorials.
* **Developers who learn by tracing**, those who want to follow not just the data, but the code paths.
* **Students and self-taught learners** who are tired of “black box” usage and want to go under the hood.
* **Practitioners using `scikit-learn` in real-world projects** who want to deeply understand what each pipeline component is actually doing.

If you’ve ever:

* Wondered what happens when you call `.fit()` or `.transform()`,
* Wanted to visualize an algorithm before trusting it,
* Or felt that ML hides too much beneath “user-friendly” layers—

This book is for you.

---

### From Abstraction to Understanding: How This Book Was Born

The turning point for me was simple but powerful:

> “I want to understand machine learning the way I debug firmware.”

That meant stepping through the process line by line, understanding input/output, tracing the flow of logic, and visualizing how each algorithm operates—not just in math, but in code.

I started with projects like fraud detection and classification tasks. I copied tutorials. I ran pipelines. But I didn’t *feel* like I understood anything. So I began slowing down.

I asked:

* What exactly is `StandardScaler` computing?
* What does `SVC(probability=True)` change internally?
* Why does `CalibratedClassifierCV` improve thresholding?
* How does `SMOTE` actually synthesize new data?

These weren’t just academic questions. They were **engineering questions**.

And every time I found an answer, I wrote it down—in plain English, with code, math, and visuals.

Now, those answers form the chapters of this book.

---

### What You’ll Learn (and What You Won’t)

You will learn:

* What classical ML algorithms like **Logistic Regression**, **SVM**, **Decision Trees**, **Random Forests**, and **KNN** do, mathematically and in code.
* How **scikit-learn** is designed, from `fit`/`transform` to pipelines and grid searches.
* How to perform **data preprocessing**, **model evaluation**, and **threshold tuning** correctly.
* How to read and interpret **metrics** (accuracy, F1, PR-AUC) in imbalanced datasets.
* How to **trace behavior**, **analyze model decisions**, and **debug performance issues**.

You will *not* find:

* Generic introductions to AI or deep learning (this book is about classical ML).
* Fluffy motivational writing that avoids technical rigor.
* Opaque math without context or practical application.

This is a **hands-on, traceable, engineer-friendly deep dive** into how classical machine learning works, with `scikit-learn` as our laboratory.

---

### How to Read This Book

Each chapter is built to help you understand both **why** and **how** machine learning algorithms work.

* You can read sequentially, or jump to specific models or tools.
* Diagrams and simplified math help visualize what’s happening under the hood.
* Code examples use real datasets and follow best practices for **evaluating, tuning, and interpreting models**.
* Bonus “source trace” boxes show what `scikit-learn` is actually doing in key functions.
* Appendices include resources for further reading, math refreshers, and command cheat sheets.

If you’ve ever felt like machine learning is a set of magical boxes, this book is here to **open those boxes**, and show you what’s really inside.

This is not the book that teaches you how to pass an ML interview.
This is the book that helps you finally say:

> “Now I understand how that works—inside and out.”

Let’s begin.

---