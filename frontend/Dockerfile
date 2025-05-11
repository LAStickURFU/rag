FROM node:18-alpine as build

WORKDIR /app

# Копирование package.json и package-lock.json
COPY package*.json ./

# Установка зависимостей
RUN npm ci

# Копирование исходного кода
COPY . .

# Сборка приложения
RUN npm run build

# Настройка рабочего окружения
FROM nginx:alpine

# Копирование собранного приложения
COPY --from=build /app/build /usr/share/nginx/html

# Копирование конфигурации NGINX 
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"] 