/**
 * 控制台日志美化工具（支持 Node.js 环境）
 */

// ANSI 转义颜色码（仅在支持终端颜色的环境中生效）
export const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  underscore: '\x1b[4m',
  blink: '\x1b[5m',
  reverse: '\x1b[7m',
  hidden: '\x1b[8m',

  fg: {
    black: '\x1b[30m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m',
    white: '\x1b[37m',
    // crimson 通常需要 256 色支持，这里保留但建议慎用
    crimson: '\x1b[38;5;196m', // 修正为标准 ANSI 256 色（可选）
  },

  bg: {
    black: '\x1b[40m',
    red: '\x1b[41m',
    green: '\x1b[42m',
    yellow: '\x1b[43m',
    blue: '\x1b[44m',
    magenta: '\x1b[45m',
    cyan: '\x1b[46m',
    white: '\x1b[47m',
    crimson: '\x1b[48;5;196m',
  },
} as const;

// 平台常量（提升可读性和类型安全）
export type BuildPlatform = 'h5' | 'mp-weixin' | 'app' | string;
const H5_PLATFORM = 'h5';

/**
 * 安全地拼接带颜色的字符串（自动追加 reset）
 */
const colorize = (text: string, ...styles: string[]): string => {
  return `${styles.join('')}${text}${colors.reset}`;
};

/**
 * 打印带颜色的分隔线
 */
export const printSeparator = (): void => {
  console.log(colorize('═════════════════════════════════════════════════════════', colors.bright, colors.fg.cyan));
};

/**
 * 打印带颜色的标签行
 * @param label 标签名称
 * @param value 值（支持 string | number）
 */
export const printLabelValue = (label: string, value: string | number): void => {
  const labelPart = colorize(`• ${label}:`, colors.bright, colors.fg.green);
  const valuePart = colorize(String(value), colors.fg.yellow);
  console.log(`${labelPart} ${valuePart}`);
};

/**
 * 构建环境信息所需字段接口
 */
interface BuildEnv {
  VITE_APP_TITLE?: string;
  VITE_APP_PORT?: string;
  VITE_SERVER_BASEURL?: string;
}

/**
 * 打印构建环境信息
 * @param command 命令类型（如 'build', 'dev'）
 * @param mode 环境模式（如 'development', 'production'）
 * @param platform 目标平台
 * @param env 环境变量对象
 */
export const printBuildInfo = (
  command: string,
  mode: string,
  platform: BuildPlatform,
  env: BuildEnv,
): void => {
  const { VITE_APP_TITLE = '未设置', VITE_APP_PORT, VITE_SERVER_BASEURL = '未设置' } = env;

  printSeparator();
  printLabelValue('构建模式', `${command} ${mode}`);
  printLabelValue('目标平台', platform);
  printLabelValue('应用标题', VITE_APP_TITLE);

  // 仅在 H5 平台显示端口
  if (platform === H5_PLATFORM && VITE_APP_PORT) {
    printLabelValue('服务端口', VITE_APP_PORT);
  }

  printLabelValue('API地址', VITE_SERVER_BASEURL);
  printSeparator();
};