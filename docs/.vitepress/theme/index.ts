// https://vitepress.dev/guide/custom-theme
import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import './style/index.css'
import BlogList from './components/BlogList.vue'
import Breadcrumb from './components/Breadcrumb.vue'
import DraftMessage from './components/DraftMessage.vue'
import Tag from './components/Tag.vue'
import Image from './components/Image.vue'
import ArticleStats from './components/ArticleStats.vue'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      'doc-before': () => [
        h(Breadcrumb),
        h(Tag),
        h(ArticleStats),
        h(DraftMessage)
      ]
    })
  },
  enhanceApp({ app, router, siteData }) {
    app.component('Breadcrumb', Breadcrumb) 
    app.component('BlogList', BlogList)
    app.component('Image', Image)
    app.component('ArticleStats', ArticleStats)
    // ...
  }
} satisfies Theme
