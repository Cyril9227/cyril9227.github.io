---
title: Coding stuff
permalink: /coding/
layout: page
excerpt: Random bad attempts at coding stuff
comments: false
---

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
          <div class="all-posts-tags" style="margin-left: 25px;">
            {% for tag in post.tags %}
              <a href="/tags#{{ tag | slugify }}" class="tag-item">{{ tag }}</a>
            {% endfor %}
          </div>
        </article>
      {% endfor %}
    {% else %}
      <p>This lazy bum hasn't posted anything yet...</p>
    {% endif %}
  </section>
</div>