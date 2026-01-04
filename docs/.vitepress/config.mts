import { defineConfig } from 'vitepress'
import { fileURLToPath, URL } from 'node:url'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  // 指定源文件目录，所有 Markdown 文件应放在此目录下
  srcDir: "docs",
  // 站点标题，显示在页面左上角和 meta 标签中
  title: "Peter Wang",
  // 站点描述，用于 meta 标签中的描述信息
  description: "Personal Blog",
  // 支持 math 的latex
  markdown: {
    math: true,
    image: {
      lazyLoading: true,
    },
    lineNumbers: true,
  },

  // vite
  vite:{
    resolve: {
      alias: {
        '@src': fileURLToPath(new URL('../src', import.meta.url)),
      },
    }
  },

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
    ],
    search: {
      provider: 'local'
    },
    outline: [2, 3],
    footer: {
      message:
        'ICP'
    }
  }
})
