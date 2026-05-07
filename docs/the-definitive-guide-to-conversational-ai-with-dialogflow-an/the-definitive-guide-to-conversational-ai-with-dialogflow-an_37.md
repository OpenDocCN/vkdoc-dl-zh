# ServerName 指令用于设置服务器用于标识自身的请求方案、主机名和端口。这在创建重定向 URL 时使用。在虚拟主机的上下文中，ServerName 指定了请求的 Host: 标头中必须出现的主机名，以匹配此虚拟主机。对于默认虚拟主机（此文件），此值并非决定性因素，因为它被用作最后的主机。但是，你必须为任何其他虚拟主机显式设置它。
#ServerName www.example.com
`ServerAdmin webmaster@localhost`
#DocumentRoot /var/www/projects

