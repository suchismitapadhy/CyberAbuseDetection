(function($) {
    $(function() {
        $(".button-collapse").sideNav();
        $(".parallax").parallax();
    }); // end of document ready
})(jQuery); // end of jQuery name space

$("#submit").on("click", function() {
    $(".check-for-abuse").submit()
})
