import { defineConfig } from 'astro/config';

// Deployed at https://shravanvinayhegde.github.io/Predictive-Maintenance-PyTorch/
export default defineConfig({
  site: 'https://shravanvinayhegde.github.io',
  base: '/Predictive-Maintenance-PyTorch',
  trailingSlash: 'always',
  build: {
    assets: 'assets'
  }
});
