# Component Viewer App

This is a [Next.js](https://nextjs.org) application designed to render and display TSX and HTML files from an external components directory. It was created as part of interview preparation for a Gen AI Data Scientist position at American Airlines.

The application allows you to browse, view, and render components from a specified directory, making it easy to showcase and test UI components.

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Using the Component Viewer

This application is designed to render TSX and HTML files from the `components` directory located at the same level as the application directory. Here's how to use it:

1. Place your TSX and HTML files in the `../components` directory (relative to this application)
2. Start the development server with `npm run dev`
3. Open [http://localhost:3000](http://localhost:3000) in your browser
4. Browse the list of available components
5. Click on any component to view its source code and rendered output

### Supported File Types

- **TSX Files**: React components written in TypeScript JSX
- **HTML Files**: Standard HTML files

### How It Works

The application:

1. Scans the `../components` directory for TSX and HTML files
2. Creates a dynamic page for each component
3. Displays the source code alongside the rendered component
4. For TSX files, it attempts to dynamically import and render the component
5. For HTML files, it renders the HTML content directly

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
