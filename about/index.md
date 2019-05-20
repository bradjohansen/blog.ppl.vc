---
layout: post
title: About
image: /images/about/1.jpg
---

I am a Ph.D. student in machine learning at Università della Svizzera Italiana (Lugano, Switzerland).
My research focuses on predicting and controlling anomalies on graphs. 
<br>
I mostly work on graph neural networks, reinforcement learning, and epilepsy. 

<br>
<br>

I like getting lost in nature, travelling around by bike, playing the guitar, and creating websites/pics/music.  
<br>
Power places: Eivissa | Kyōto | Queensland.

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
    <br><span style='font-size: 14px;'>ETH: <a href="0x8c4641e3EEdd08D5975cb2B5c67650eeD6B03656">0x8c4641e3EEdd08D5975cb2B5c67650eeD6B03656</a></span>
</center>

<br>

<center class="image-grid">
    <img src="/images/about/1.jpg" style="grid-column: 1 / span 2;">
    <img src="/images/about/2.jpg" style="grid-column: 1; overflow:hidden;">
    <img src="/images/about/3.jpg" style="grid-column: 2;">
    <img src="/images/about/4.jpg" style="grid-column: 1 / span 2;">
</center>

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
