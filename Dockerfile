FROM node:22-alpine

WORKDIR /app

COPY f1-dashboard/package*.json ./
RUN npm ci

COPY f1-dashboard/ ./
RUN npm run build

EXPOSE 8502

ENV NODE_ENV=production
ENV HOSTNAME=0.0.0.0
ENV PORT=8502

CMD ["npm", "start"]
