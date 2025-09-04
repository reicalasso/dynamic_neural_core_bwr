# frontend.Dockerfile
FROM node:16-alpine

WORKDIR /app

# Copy package.json and package-lock.json
COPY frontend/app/package.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend application
COPY frontend/app ./

EXPOSE 3000

# Start the Next.js dev server
CMD ["npm", "run", "dev"]
