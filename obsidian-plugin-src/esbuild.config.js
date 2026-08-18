module.exports = {
  entry: './main.ts',
  outfile: 'main.js',
  bundle: true,
  platform: 'node',
  target: 'ES6',
  format: 'cjs',
  sourcemap: 'inline',
  external: ['obsidian'],
};
