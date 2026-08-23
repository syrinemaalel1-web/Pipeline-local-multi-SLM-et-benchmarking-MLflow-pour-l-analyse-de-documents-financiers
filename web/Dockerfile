# Étape 1 : compile le site React (Vite) en fichiers statiques.
FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Étape 2 : sert les fichiers compilés avec nginx — l'image finale ne contient
# ni Node ni les sources, juste le résultat de `npm run build` (dist/).
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
