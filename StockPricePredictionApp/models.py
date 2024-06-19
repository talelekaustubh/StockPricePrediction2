from django.db import models

# Create your models here.
class StockUser(models.Model):
    name=models.CharField(name='name',max_length=50)
    email=models.EmailField(name='email', max_length=254)
    password=models.CharField(name='password', max_length=50)
    flag=models.IntegerField(name='flag')

    def __str__(self) :
        return self.name