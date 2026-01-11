/**
 * Store 模块统一导出
 * 方便外部导入使用
 */

export { useUserStore, USER_STORE_ID } from './user'
export type { UserStore } from './user'

export { useAppStore, APP_STORE_ID } from './app'
export type { AppStore } from './app'

export { usePrivacyStore, PRIVACY_STORE_ID } from './privacy'
export type { PrivacyStore } from './privacy'

// 导出示例(可选,用于学习和参考)
export * as examples from './examples'


