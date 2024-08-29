---
title: Maths stuff
permalink: /maths/
layout: page
excerpt: Random bad maths and cool problems
comments: false
---



# All Maths posts below : 

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
          <div class="all-posts-tags" style="margin-left: 25px;">
            {% for tag in post.tags %}
              <a href="/tags#{{ tag | slugify }}" class="tag-item">{{ tag }}</a>
            {% endfor %}
          </div>
        </article>
      {% endfor %}
    {% else %}
      <p>No maths posts yet...</p>
    {% endif %}
  </section>
</div>