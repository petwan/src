---
title: 🦖 微信小程序开发记录
date: 2026-01-07
tags: [个人开发]
description: 
draft: false
---

# 🦖 微信小程序开发记录

使用 UniApp + Vue3 + TypeScript 开发微信小程序，借助的是这个[模板](https://github.com/vue-rookie/uni-vue3)


## 1. 代码结构
- **api**: 保存后端请求
- **components**: 存放组件
- **hooks**：存放自定义的函数
- **pages**: 存放页面
- **static**: 存放静态资源
- **store**: 存放状态管理，针对微小程序，就是localStorage中的内容管理，key-value + 一些基础的CRUD方法
- **types**: 全局的types
- **utils**: 存放工具函数，例如 logger 和拦截器等

另外，部分的环境变量通过env进行配置。

