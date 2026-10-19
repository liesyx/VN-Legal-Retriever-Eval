# Giao diện tĩnh: busybox httpd (~1 MB), chạy non-root, cổng 8080. Trang gọi backend legal-api /ask.
FROM busybox:1.36.1-musl
COPY index.html styles.css app.js /www/
COPY httpd.conf /etc/httpd.conf
USER 65534:65534
EXPOSE 8080
CMD ["httpd", "-f", "-v", "-p", "8080", "-h", "/www", "-c", "/etc/httpd.conf"]
