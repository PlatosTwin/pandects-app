# Proxy to MariaDB machine on Fly.io
```
flyctl proxy 13306:3306 -a pandects-db  
```

# Dump DB (local)
```                                                  
mysqldump -u root -p mna agreements, llm_output, prompts, taxonomy, sections xml > tables.sql
```

# Upload local dump to MariaDB machine on Fly.io
```
mysql -u root -p -h 127.0.0.1 -P 13306 mna < tables.sql
```