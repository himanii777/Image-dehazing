I wanted to change the previous implementation because it kept giving me contrasted images (even though I changed the parameters)
Usually when we can lower the contrast, dehazing doesn’t happen that well. So for varying haze intensities, I implemented multi scaled dark channel estimation.

For the same paramter, comparison: 

Multiscale

![image](https://github.com/user-attachments/assets/17d624a4-ddb3-45cf-8a78-fd74f5ad695b)

Original

![image](https://github.com/user-attachments/assets/fe46a1bd-39ff-42d7-9c4f-cbba7cfe9cb1)

