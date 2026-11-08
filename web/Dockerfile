# Giao diện tĩnh: busybox httpd (~1 MB), chạy non-root, cổng 8080. Trang gọi backend legal-api /ask.
FROM busybox:1.36.1-musl
ARG VERSION=dev
COPY index.html styles.css app.js /www/
# Gắn số build vào đường dẫn css/js -> mỗi lần deploy trình duyệt tải file mới, không dùng bản cache cũ
RUN sed -i "s#href=\"styles.css\"#href=\"styles.css?v=${VERSION}\"#; s#src=\"app.js\"#src=\"app.js?v=${VERSION}\"#" /www/index.html \
 && grep -q "app.js?v=${VERSION}" /www/index.html
COPY httpd.conf /etc/httpd.conf
USER 65534:65534
EXPOSE 8080
CMD ["httpd", "-f", "-v", "-p", "8080", "-h", "/www", "-c", "/etc/httpd.conf"]
