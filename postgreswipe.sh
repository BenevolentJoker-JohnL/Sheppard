sudo systemctl stop postgresql
sudo apt-get --purge remove postgresql postgresql-*
sudo apt-get --purge remove postgresql-client-*
sudo apt-get --purge remove postgresql-client-common postgresql-common
sudo rm -rf /var/lib/postgresql/
sudo rm -rf /var/log/postgresql/
sudo rm -rf /etc/postgresql/
sudo rm -rf /etc/postgresql-common/
sudo deluser postgres
sudo rm -rf /home/postgres
sudo userdel -r postgres
sudo apt autoremove
sudo apt clean
