#!/bin/bash
(test -e rsa/private.pem && echo 'Keys already there') || (openssl genrsa -out rsa/private.pem 2048 && openssl rsa -in rsa/private.pem -outform PEM -pubout -out rsa/public.pem)
