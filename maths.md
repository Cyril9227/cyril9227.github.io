---
title: Maths stuff
permalink: /maths/
layout: page
excerpt: Random bad maths and cool problems
comments: false
---


<div id="posts-container">
  <section class="tag-section">
    {% assign maths_posts = site.tags.maths %}
    {% if maths_posts.size > 0 %}
      {% for post in maths_posts %}
        <article class="post-item">
          <span class="post-item-date">{{ post.date | date: "%b %d, %Y" }}</span>
          <h3 class="post-item-title">
            <a href="{{ post.url }}">{{ post.title | escape }}</a>
          </h3>
          <span class="post-item-tags">
            {% for tag in post.tags %}
              <a href="/tags#{{ tag | slugify }}" class="tag-item">{{ tag }}</a>
            {% endfor %}
          </span>
        </article>
      {% endfor %}
    {% else %}
      <p>No mathematics posts found.</p>
    {% endif %}
  </section>
</div>