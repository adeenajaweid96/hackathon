---
sidebar_position: 85
title: "Deployment Configuration"
---

# Deployment Configuration

## GitHub Pages Deployment

This guide covers the deployment configuration for hosting the Physical AI & Humanoid Robotics book on GitHub Pages.

### Prerequisites

- GitHub repository with the book code
- GitHub Actions enabled for the repository
- Properly configured `baseUrl` in `docusaurus.config.ts`

### Configuration

#### Docusaurus Configuration

Ensure your `docusaurus.config.ts` has the correct GitHub Pages settings:

```typescript
const config = {
  // ...
  url: 'https://your-username.github.io',
  baseUrl: '/your-repository-name/',
  organizationName: 'your-username',
  projectName: 'your-repository-name',
  // ...
};
```

#### GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  deploy:
    name: Deploy to GitHub Pages
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
          cache: npm

      - name: Install dependencies
        run: cd book && npm ci

      - name: Build website
        run: cd book && npm run build

      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./book/build
          user_name: github-actions[bot]
          user_email: 41898282+github-actions[bot]@users.noreply.github.com
```

### Deployment Steps

1. **Prepare your repository**:
   - Ensure the book code is in the main branch
   - Verify the `baseUrl` matches your repository name
   - Test the build locally with `npm run build`

2. **Enable GitHub Pages**:
   - Go to your repository settings
   - Navigate to the "Pages" section
   - Select "GitHub Actions" as the source
   - Save the settings

3. **Trigger deployment**:
   - Push changes to the main branch
   - GitHub Actions will automatically build and deploy
   - Monitor the Actions tab for deployment status

### Custom Domain Configuration

If using a custom domain:

1. Add a `CNAME` file to the `static` directory:
   ```
   your-domain.com
   ```

2. Configure the domain in GitHub repository settings under "Pages"

### Environment Variables

For the backend API integration, set environment variables in GitHub Secrets:

- `API_BASE_URL`: Base URL for the backend API
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

### Performance Optimization for Deployment

#### Asset Optimization
- Enable gzip compression
- Use image optimization
- Minify CSS and JavaScript
- Implement proper caching headers

#### SEO Configuration
- Add meta tags and descriptions
- Configure social media cards
- Implement sitemap generation
- Set up Google Analytics

### Monitoring Deployment

#### Health Checks
- Monitor page load times
- Check for broken links
- Verify API connectivity
- Monitor error rates

#### Analytics
- Set up Google Analytics
- Track user engagement
- Monitor popular pages
- Analyze user behavior

### Troubleshooting

#### Common Issues
- **404 Errors**: Verify `baseUrl` configuration
- **CSS Not Loading**: Check asset paths in production
- **Links Not Working**: Ensure relative paths are correct
- **Images Not Showing**: Verify image paths and formats

#### Debugging Steps
1. Check the browser console for errors
2. Verify the build process locally
3. Review GitHub Actions logs
4. Test on different browsers and devices

### Rollback Procedures

If a deployment causes issues:

1. Identify the problematic commit
2. Create a new branch from the previous working commit
3. Deploy the working version
4. Fix the issues in a separate branch
5. Deploy the fixed version once verified

### Continuous Integration

#### Automated Testing
- Unit tests for components
- Integration tests for pages
- Accessibility tests
- Performance tests

#### Code Quality
- Linting checks
- Type checking
- Security scanning
- Accessibility auditing

This deployment configuration ensures that the Physical AI & Humanoid Robotics book is deployed efficiently and reliably to GitHub Pages with proper monitoring and maintenance procedures.