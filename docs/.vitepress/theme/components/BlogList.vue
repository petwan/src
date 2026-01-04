<!-- BlogList.vue -->
<script setup>
import { ref, computed } from 'vue'
import BlogPost from './BlogPost.vue'

// 使用 Vite 的 glob 导入所有 blog 下的 .md 文件（含 frontmatter）
const postsModules = import.meta.glob('/blog/**/*.md', { eager: true })

// 所有文章数据
const blogPosts = ref([])
// 当前选中的标签（null 表示不过滤）
const selectedTag = ref(null)

// 提取文章数据
function extractPosts() {
    const posts = []

    for (const [path, module] of Object.entries(postsModules)) {
        const fm = module.__pageData?.frontmatter
        if (!fm || fm.draft === true) continue
        if (path.endsWith('/index.md')) continue

        posts.push({
            url: path.replace(/^\/blog/, '/blog').replace(/\.md$/, ''),
            title: fm.title || 'Untitled',
            date: fm.date ? new Date(fm.date).toISOString().split('T')[0] : '1970',
            description: fm.description || fm.excerpt || '',
            tags: Array.isArray(fm.tags) ? fm.tags : [],
            author: fm.author || ''
        })
    }

    // 按日期倒序排序
    posts.sort((a, b) => new Date(b.date) - new Date(a.date))
    blogPosts.value = posts
}

// 提取数据（静态构建时执行）
extractPosts()

// 获取所有唯一标签（用于筛选栏）
const allTags = computed(() => {
    const tagSet = new Set()
    blogPosts.value.forEach(post => {
        post.tags.forEach(tag => tagSet.add(tag))
    })
    return Array.from(tagSet).sort()
})

// 计算过滤后的文章列表
const filteredPosts = computed(() => {
    if (!selectedTag.value) {
        return blogPosts.value
    }
    return blogPosts.value.filter(post =>
        post.tags.includes(selectedTag.value)
    )
})

// 切换标签筛选
function selectTag(tag) {
    selectedTag.value = selectedTag.value === tag ? null : tag
}

// 清除筛选
function clearFilter() {
    selectedTag.value = null
}
</script>

<template>
    <div class="blog-list">
        <!-- 标签筛选区 -->
        <div v-if="allTags.length > 0" class="tag-filter">
            <span class="filter-label">筛选标签：</span>
            <button v-for="tag in allTags" :key="tag" @click="selectTag(tag)" :class="{ active: selectedTag === tag }"
                class="tag-button">
                {{ tag }}
            </button>
            <button v-if="selectedTag" @click="clearFilter" class="clear-button">
                × 清除
            </button>
        </div>

        <!-- 博客文章列表 -->
        <BlogPost v-for="post in filteredPosts" :key="post.url" :title="post.title" :url="post.url" :date="post.date"
            :description="post.description" :tags="post.tags" :author="post.author" />

        <div v-if="filteredPosts.length === 0" class="no-posts">
            暂无符合条件的博客文章。
        </div>
    </div>
</template>

<style scoped>
.blog-list {
    margin: 0 auto;
    padding: 0 1.5rem;
    max-width: 100%;
}

.tag-filter {
    margin-bottom: 2rem;
    padding: 0.5rem 0;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
}

.filter-label {
    font-weight: 600;
    color: var(--vp-c-text-1);
}

.tag-button {
    padding: 0.25rem 0.75rem;
    border: 1px solid var(--vp-c-border);
    border-radius: 4px;
    background: transparent;
    color: var(--vp-c-text-2);
    cursor: pointer;
    transition: all 0.2s;
}

.tag-button:hover {
    color: var(--vp-c-text-1);
    border-color: var(--vp-c-brand);
}

.tag-button.active {
    background: var(--vp-c-brand);
    color: white;
    border-color: var(--vp-c-brand);
}

.clear-button {
    padding: 0.25rem 0.75rem;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    background: #f5f5f5;
    color: #666;
    cursor: pointer;
    font-size: 0.9em;
}

.no-posts {
    text-align: center;
    color: var(--vp-c-text-2);
    padding: 3rem 0;
}
</style>