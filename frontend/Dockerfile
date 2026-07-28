# Build stage
FROM node:20-slim AS build

WORKDIR /app

# Copy package management files
COPY package*.json ./
RUN npm ci

# Copy files and build application
COPY . .
RUN npm run build

# Production server stage (using Nginx)
FROM nginx:alpine

# Copy production build files to Nginx public folder
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
