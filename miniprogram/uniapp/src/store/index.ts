/**
 * Pinia Store 配置和导出
 * 提供状态管理核心功能
 */

import { createPinia } from 'pinia'
import { createPersistedState } from 'pinia-plugin-persistedstate'

// ==================== 创建 Pinia 实例 ====================
const store = createPinia()

// ==================== 配置持久化插件 ====================
store.use(
  createPersistedState({
    // 自定义 storage 对象,替换默认的 localStorage
    storage: {
      // 读取数据的方法:使用 uni.getStorageSync 同步获取本地存储
      getItem(key: string) {
        return uni.getStorageSync(key)
      },
      // 写入数据的方法:使用 uni.setStorageSync 同步设置本地存储
      setItem(key: string, value: string) {
        uni.setStorageSync(key, value)
      },
    },
  }),
)

// ==================== 导出 ====================
export default store

// 导出所有模块
export * from './modules'

// 导出类型（仅在类型层面导出，不会生成运行时代码）
export type * from './types'

