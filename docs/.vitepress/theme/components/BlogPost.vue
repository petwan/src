<template>
    <article class="blog-post">
        <div class="post-header">
            <h3 class="post-title">
                <a :href="url">{{ title }}</a>
            </h3>
            <div class="post-meta">
                <time :datetime="date" class="post-date">
                    {{ formatDate(date) }}
                </time>
            </div>
        </div>

        <div class="post-content">
            <div class="post-text">
                <p class="post-description">{{ description }}</p>

                <div class="post-footer">
                    <div class="post-tags" v-if="tags?.length">
                        <strong>Tags:</strong>
                        <code v-for="tag in tags" :key="tag" class="post-tag">{{
                            tag
                        }}</code>
                    </div>
                    <a :href="url" class="read-more"> Read more → </a>
                </div>
            </div>
        </div>
    </article>
</template>

<script setup>
const props = defineProps({
    title: String,
    url: String,
    date: String,
    author: String,
    description: String,
    tags: Array,
    image: String
});

function formatDate(date) {
    return new Date(date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}
</script>

<style scoped>
.blog-post {
    display: block;
    text-decoration: none;
    color: var(--vp-c-text-1);
    background: var(--vp-c-bg-soft);
    border: 1px solid var(--vp-c-divider);
    border-radius: 12px;
    padding: 16px;
    transition: all 0.2s ease;
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    margin-bottom: 16px;
}

.blog-post:hover {
    border-color: var(--vp-c-brand);
    background: var(--vp-c-bg-mute);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
}

.post-title {
    margin: 0 0 0.2rem 0;
}

.post-title a {
    transition: color 0.2s;
}

.post-title a:hover {
    color: var(--vp-c-brand-1);
}

.post-date {
    color: var(--vp-c-text-2);
    font-size: 0.9rem;
}

.post-content {
    display: flex;
    gap: 2rem;
    align-items: flex-start;
}

.post-image {
    flex-shrink: 0;
    width: 300px;
}

.post-image img {
    width: 100%;
    height: auto;
    border-radius: 8px;
    object-fit: cover;
    aspect-ratio: 16 / 9;
}

.post-text {
    flex: 1;
}

.post-description {
    color: var(--vp-c-text-2);
    line-height: 1.6;
    margin: 1.5rem 0;
    font-size: 1.05rem;
}

.post-footer {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    margin-top: 1.5rem;
    flex-wrap: wrap;
    gap: 1rem;
}

.post-tags {
    color: var(--vp-c-text-2);
    font-size: 0.9rem;
}

.post-tag {
    background: var(--vp-c-default-soft);
    color: var(--vp-c-text-2);
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    margin-left: 0.5rem;
    font-family: var(--vp-font-family-mono);
}


/* Responsive */
@media (max-width: 768px) {
    .post-content {
        flex-direction: column;
        gap: 1rem;
    }

    .post-image {
        width: 100%;
    }

    .post-title {
        font-size: 1.5rem;
    }

    .post-footer {
        flex-direction: column;
        align-items: flex-start;
    }
}
</style>