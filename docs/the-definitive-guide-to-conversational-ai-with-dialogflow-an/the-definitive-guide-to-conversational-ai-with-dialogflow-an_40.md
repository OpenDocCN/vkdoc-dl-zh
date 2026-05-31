# 也可以为特定模块配置日志级别，例如：
`LogLevel info ssl:debug`
`ErrorLog /var/www/projects/error.log`
`CustomLog /var/www/projects/access.log combined`
`ServerName mtls.conv.dev`
`SSLCertificateFile /etc/letsencrypt/live/mtls.conv.dev/fullchain.pem`
`SSLCertificateKeyFile /etc/letsencrypt/live/mtls.conv.dev/privkey.pem`
`Include /etc/letsencrypt/options-ssl-apache.conf`
`SSLVerifyClient require`
`SSLVerifyDepth 2`
`SSLCACertificateFile "/var/www/projects/ca-crt.pem"`

