---
layout: post
title: About
---
<script src="//ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>
<script>window.jQuery || document.write('<script src="_/js/libs/jquery-1.9.1.min.js"><\/script>')</script>

<p>
I am a Master's Student of Computer Science and Engineering at Politecnico di Milano, with a specialization and a passion for AI, machine learning, and recommendation systems. 
</p>

<p>
My work mostly focuses on deep reinforcement learning and high performance recommendation systems, but I try to keep my horizon as wide as possible.
I enjoy reading, biking, playing the ukulele and traveling the world whenever I have the chance. 
</p>

<p>
My second home is in Ibiza, where I go every summer to get lost in nature and find myself again.
</p>

<p>
The purpose of this website is to leave a trace of the many ideas that inhabit my mind, in the hope of inspiring other people to think and share new amazing ideas so that we can all work together towards a better future.
</p>

<center>
    <div class="photoset-grid-custom" data-layout="121">
        <img src="/images/about/1.jpg">
        <img src="/images/about/2.jpg">
        <img src="/images/about/3.jpg">
        <img src="/images/about/4.jpg">
    </div>
</center>

<script src="/assets/js/jquery.photoset-grid.js"></script>

<script type="text/javascript">
    $('.photoset-grid-custom').photosetGrid({
    // Set the gutter between columns and rows
    gutter: '5px',
  
    // Wrap the images in links
    highresLinks: true,
  
    // Asign a common rel attribute
    rel: 'print-gallery',

    onInit: function(){},
    
    onComplete: function(){
        // Show the grid after it renders
        $('.photoset-grid-custom').attr('style', '');
    }
});
</script>

