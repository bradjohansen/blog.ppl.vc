---
layout: post
title: About
---

<p>
I am a PhD Student of Computer Science at Università della Svizzera Italiana 
(USI), graduated in Computer Science and Engineering at Politecnico di Milano in
 2017.
My work mostly focuses on deep learning, reinforcement learning, and recommender 
systems.
</p>

<p>
I am an avid reader, bicycle revolution supporter, ukulele player, red cat owner, 
and worldwide traveler. 
<br>
My second home is in Ibiza, where I go every summer to get lost in nature and find myself again.
</p>

<span>The purpose of this website is to leave a trace of my ideas, my work, and my experiences.</span>

<br>
<br>

<center>
    <span class='personal-social-media'>
        <a target="_blank" href="https://twitter.com/riceasphait">
            <img class="svg" src="/assets/icons/twitter.svg" width="30" height="30">
        </a>
        <a target="_blank" href="https://github.com/danielegrattarola">
            <img class="svg" src="/assets/icons/github.svg" width="30" height="30">
        </a>
        </a>
        <a target="_blank" href="https://linkedin.com/in/danielegrattarola">
            <img class="svg" src="/assets/icons/linkedin.svg" width="30" height="30">
        </a>
        <a target="_blank" href="/feed.xml">
            <img class="svg" src="/assets/icons/rss.svg" width="30" height="30">
        </a>
    </span>
    <br><a target="_blank" href="/files/Daniele_Grattarola_resume.pdf">Résumé</a>
    <br><span style='font-size: 14px;'>BTC address: <a href="bitcoin:1KLyYmceVcs1kXWkUcwrQBGrZymc85scXa">1KLyYmceVcs1kXWkUcwrQBGrZymc85scXa</a></span>
</center>

<br>

<center>
    <div class="photoset-grid-custom" data-layout="121">
        <img src="/images/about/1.jpg">
        <img src="/images/about/2.jpg">
        <img src="/images/about/3.jpg">
        <img src="/images/about/4.jpg">
    </div>
</center>


<!-- PHOTOSET GRID -->
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

<!-- SVG-->
<script type="text/javascript">
/*
 * Replace all SVG images with inline SVG
 */
jQuery('img.svg').each(function(){
    var $img = jQuery(this);
    var imgID = $img.attr('id');
    var imgClass = $img.attr('class');
    var imgURL = $img.attr('src');

    jQuery.get(imgURL, function(data) {
        // Get the SVG tag, ignore the rest
        var $svg = jQuery(data).find('svg');

        // Add replaced image's ID to the new SVG
        if(typeof imgID !== 'undefined') {
            $svg = $svg.attr('id', imgID);
        }
        // Add replaced image's classes to the new SVG
        if(typeof imgClass !== 'undefined') {
            $svg = $svg.attr('class', imgClass+' replaced-svg');
        }

        // Remove any invalid XML tags as per http://validator.w3.org
        $svg = $svg.removeAttr('xmlns:a');

        // Replace image with new SVG
        $img.replaceWith($svg);

    }, 'xml');

});

</script>
