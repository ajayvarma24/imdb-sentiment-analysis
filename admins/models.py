from django.db import models

class InputData(models.Model):
    text = models.TextField()

    def __str__(self):
        return self.text
