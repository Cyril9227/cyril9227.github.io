---
title: Coding stuff
permalink: /coding/
layout: page
excerpt: Random bad attempts at coding stuff
comments: false
---

# HELLO WORLD

<div id="posts-container">
  <section class="tag-section">
    {% assign coding_posts = site.tags.coding %}
    {% if coding_posts.size > 0 %}
      {% for post in coding_posts %}
        <article class="post-item">
          <span class="post-item-date">{{ post.date | date: "%b %d, %Y" }}</span>
          <h3 class="post-item-title">
            <a href="{{ post.url }}">{{ post.title | escape }}</a>
          </h3>
        </article>
      {% endfor %}
    {% else %}
      <p>Oh oh seems that this lazy fck didn't publish anything yet...</p>
    {% endif %}
  </section>
</div>