---
title: "How to steal my heart (and approval): A code reviewer's dream"
date: 2023-10-14 08:38:03 +00:00
modified: 2024-05-04 10:30:02 +00:00
tags: [tech]
description: What, why and how to review changed code.
toc: true
---
Before diving into the topic of code reviews, I will be very presumptuous and make a ridiculous assumption: **we need to live**. If you disagree and/or are depressed, please continue reading not because I will cure your depression, but because I list some ideas that *might* motivate you and your team to do some code reviews.

# Back to the basics
To live, we need to access some **essential resources** (e.g., water, that’s it really). We can easily access them by **paying money**. Furthermore, by paying money, we can access more things that make us comfortable and **happy**.

But **how do we earn money** exactly?

A common way to earn money is to **solve problems** where the solution is rewarded with money. 

Usually
- the *bigger* the problem is, the bigger the reward is
- the *better* (i.e., higher quality) the solution is, the bigger the reward is

So, we have 2 clearly defined **goals** now:
1. Solve bigger problems
2. Come up with better solutions

Solving bigger problems is easy: just **hire more** people! However, there is a caveat; this **decreases the reward** of the solutions. 

# Solving "bigger and better"
Hiring more people decreases the *quality* of the solution, and it's pretty hard to maintain or increase it. This problem is commonly known as the **project management triangle**[.](https://en.wikipedia.org/wiki/Project_management_triangle)

<figure>
	<img src="/how-to-steal-my-heart-and-approval-a-code-reviewers-dream/project-management-triangle-with-comments.png" alt="project management triangle with comments">
	<figcaption>The project management triangle (with comments, read the next paragraph)</figcaption>
</figure>

Imagine that we are a dot in the triangle. We may only stay in a single spot. According to this triangle, when we hire more people, the cost will increase, so the time and the amount of features it introduces will decrease.

Depending on the business model, the project managers might be more lenient on one side; however, most businesses aim to be in the middle. Luckily, there are **many examples** of companies and their practices which **proved to be successful** in solving big problems by managing many people and yet maintaining a high quality for their solutions. So, I went through some of the most money-generating big and tech businesses, and read their **Software Engineering handbooks** (e.g., [Google's handbook](https://abseil.io/resources/swe-book)). I learned how they achieve high-quality solutions with many people, and I will be sharing my understanding.

# Why review someone else code?

## The best of ~~both~~ all worlds
While trying to solve big problems, companies need higher solution output, so they hire more problem solvers. Imagine we are at one of those companies trying to solve a big problem, and have enough of good problem solvers. If we ask the same problem to 50 of them, we will (or might) get 50 different answers. 

Within those 50 answers, there will be answers which are:
- more **correct** (e.g., satisfies all acceptance criteria)
- **easier to understand/readable** (e.g., has cleaner variable/function/class names)
- more **efficient** (e.g., has lower time/space complexity)
- more **stylish** (e.g., follows the team's/company's/language's styling guides and best practices)
- more **maintainable** (e.g., causes less bugs in the future)
- more **bug-free** 
- more **tested** (e.g., has more test cases )
- more **documented**
- and more mores...
  
What if we could combine the most correct solution with the most readable, and then **combine the rest of the qualities** in the list... 

**It's possible!** At least to some extent.

We pick the most bug-free solution...
And we pick the most readable solution...
Then, we ask them to peer review each other's code...

YES! WE JUST DISCOVERED THE TERM **"CODE REVIEW"** 🥳👯🪩🥂

Due to time constraints, we can't compare every single solution to each other, but we can hire better problem solvers and make only a few review each other's code. 

Furthermore, imagine the peer review was done not just verbally, but also written. So, the changes and comments will act as a **historical record**.

## Sharing is caring

As well as achieving a high quality, code review has other benefits.

- Every engineer gets to know more of the codebase
	- More engineers can implement more solutions
	- When someone quits, not too many resources are spent relearning that particular codebase
- Decreases imposter syndrome
	- Once in a blue moon, the reviewer might see some shit code and feel good about themselves for thinking "I can do better than that"
- It makes the engineer feel a part of the (bigger) team

# How to review the code?

Surprisingly, **Google doesn't have a strict set of rules** on how to perform code reviews. They found out that when creative solutions are required, stricter environments perform worse compared to more relaxed environments.

## Dividing the workload

However, there are 3 main points that Google likes to pay specific attention to: 
- **correctness**
- **consistency**
- **language readability/code quality**

And they distribute these responsibilities to **3 roles**:

1. The first role (who is familiar with the code and passed "the process of code readability training in a particular language") approves the correctness
	<details>
	<summary>correctness <i>(click on the "▶" to expand!)</i></summary>
	<ul>
		<li><b>Correctness</b>: Is the expected behaviour equal to the current behaviour? Can I answer "Why would this solution NOT work"?</li>
			<ul>
				<li><b>Scope</b>: Is there any part of the change that is out-of-scope?</li>
				<li><b>Planning</b>: Did the author make any assumptions?</li>
				<li><b>Different parts</b>:</li>
				<ul>
					<li><b>Data structure</b>: Could we reuse existing ones? Are there any memory leaks?</li>
					<li><b>Algorithm</b>: Does it have optimal space and time complexity?</li>
					<li><b>Design</b>: Carefully inspect the design and compare it with the current result.</li>
					<li><b>Security</b>: Is any secret exposed?</li>
				</ul>
				<li><b>Outside the box</b>: Did the author consider and handle edge cases?</li>
				<li><b>Tests</b>: Does the code have automated tests that cover all the new functionality including edge cases?</li>
				<li><b>Documentation</b>: Did the developer update the relevant documentation?</li>
			</ul>
	</ul>
	</details>

2. The second role (i.e., an *owner* who is a senior and has deep knowledge of this codebase) approves that the code is appropriate for this particular part of the codebase 
	<details>
	<summary>consistency</summary>
	<ul>
		<li><b>Consistency</b>: Is there any part that can be rewritten with existing functions/variables? Answering this question requires good knowledge of the codebase.</li>
		<ul>
			<li><b>Modularisation/Maintainability</b>: Is the code flexible enough to be reused in the <i>near</i> future?</li>
		</ul>
	</ul>
	</details>

3. The third role (who is not very familiar with the particular codebase) approves the language readability
	<details>
	<summary>language readability/code quality</summary>
	<ul>
		<li><b>Language readability/code quality</b>: Can I read the diff seamlessly?</li>
		<ul>
			<li><b>Naming</b>: Did the author choose clear names for variables, classes, methods, etc.?</li>
			<li><b>Style</b>: Does the code follow our style guides (e.g., ESLint rules, Prettier rules, some of <a href="http://google.github.io/styleguide/">Google's style guide</a>)? Don't make comments on personal preference.</li>
		</ul>
	</ul>
	</details>

These roles may be done by different individuals, but usually, **the same person does all three**.

## Human shortcuts

Google introduced some **abbreviations** to speed up the code reviewing. The most popular abbreviations that are used by the reviewer are:
- **LGTM** (looks good to me): Approves the comprehensiveness and correctness of the change.
- **FYI** (for your information): No action is required, but note to the author.
- **PTAL** (please take another look): When the reviewer disagrees with the owner, they should recommend an alternative solution and ask the reviewer to take another look at their comment.

## Robot shortcuts

Google has a step before submitting the change: **presubmit**. A presubmit process is an automated process that runs **tests**, **linters**, and **formatters** before submitting the change. This helps to scale the code by reducing the workload of engineers.

## Further best practices

### When coming up with a change request...

- **Optimise for the reader**
	- Write good descriptions
	- Write the summary of the change
	- Explain "why" this was the proposed change
	- Include a demo (e.g., if it's a front end change)
- **Keep the change under 200 lines**
	- Easier to digest individually
	- Introduces fewer merging conflicts
	- Easier to find bugs
- **Review the change within 24 working hours**
	- A shipped solution is more important than a work-in-progress change
	- As an author, coming back at the code after some time working on something different can be mentally exhausting

### When reviewing a code...

- **Be polite and professional**
	- Butchering and being "brutally honest" discourages the author from providing more solutions
- **Cherish the author's solution**
	- ✅ I loved how you solved X!
- **Don't assume**
- **Ask anything that is unclear**
- **Replace commanding with asking**
	- ❌ Record a demo for god's sake!
	- ✅ Can you record a demo?
- **Replace "you" with "we"**
	- ❌ Can you record a demo?
	- ✅ Can we have a demo?
- **Don't reject a change just because the approach is different**
	- As long as the given solution is within the agreed standards, don't point out an approach just because it's your personal preference  

# Conclusion

We started the discussion with needing money to live and be happier. Then, we mentioned how to make money (e.g., solving problems), and what increases the monetary reward (e.g., the bigger the problem and the better the solution, the bigger the reward).

After finding the motivation to do bigger and better, we provided an example to find the origins of code review. Then, we went into detail on why code review is important, and the best practices that Google follows to maximise the reward.

# TL;DR

- Code review is a peer review and a historical record of that change.
- Code review improves the solutions by combining better qualities of different solutions and perspectives.
- Code review helps with sharing knowledge, decreasing imposter syndrome, and making the engineer feel part of a team.
- Avoid setting strict rules, because creative solutions require relaxed environments.
- When reviewing a code, check for the following qualities:
	<details>
	<summary>correctness <i>(click on the "▶" to expand!)</i></summary>
	<ul>
		<li><b>Correctness</b>: Is the expected behaviour equal to the current behaviour? Can I answer "Why would this solution NOT work"?</li>
			<ul>
				<li><b>Scope</b>: Is there any part of the change that is out-of-scope?</li>
				<li><b>Planning</b>: Did the author make any assumptions?</li>
				<li><b>Different parts</b>:</li>
				<ul>
					<li><b>Data structure</b>: Could we reuse existing ones? Are there any memory leaks?</li>
					<li><b>Algorithm</b>: Does it have optimal space and time complexity?</li>
					<li><b>Design</b>: Carefully inspect the design and compare it with the current result.</li>
					<li><b>Security</b>: Is any secret exposed?</li>
				</ul>
				<li><b>Outside the box</b>: Did the author consider and handle edge cases?</li>
				<li><b>Tests</b>: Does the code have automated tests that cover all the new functionality including edge cases?</li>
				<li><b>Documentation</b>: Did the developer update the relevant documentation?</li>
			</ul>
	</ul>
	</details>
	<details>
	<summary>consistency</summary>
	<ul>
		<li><b>Consistency</b>: Is there any part that can be rewritten with existing functions/variables? Answering this question requires good knowledge of the codebase.</li>
		<ul>
			<li><b>Modularisation/Maintainability</b>: Is the code flexible enough to be reused in the <i>near</i> future?</li>
		</ul>
	</ul>
	</details>
	<details>
	<summary>language readability/code quality</summary>
	<ul>
		<li><b>Language readability/code quality</b>: Can I read the diff seamlessly?</li>
		<ul>
			<li><b>Naming</b>: Did the author choose clear names for variables, classes, methods, etc.?</li>
			<li><b>Style</b>: Does the code follow our style guides (e.g., ESLint rules, Prettier rules, some of <a href="http://google.github.io/styleguide/">Google's style guide</a>)? Don't make comments on personal preference.</li>
		</ul>
	</ul>
	</details>
- Automate processes to decrease the workload of the reviewer.
- Optimise for the reader, over-communicate.
- Keep the change short, less than 200 lines.
- Review a change within 24 hours, done work is better than undone work.
- Never assume, always ask.
- Be kind.
