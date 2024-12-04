
/* borrowed from http://www.keyboardninja.eu/webdevelopment/a-simple-search-with-angularjs-and-php */
var app = angular.module("DeseretTranslatorApp", []);

app.controller("TranslationController", ['$scope', '$http', '$sce', function($scope, $http, $sce) {
	$scope.url = '/json/translation'; // The url of our search
		
	// The function that will be executed on button click (ng-click="translate()")
	$scope.englishToDeseret = function() {
		
		// Create the http post request
		// the data holds the keywords
		// The request is a JSON request.
		$http.post($scope.url, { "english" : $scope.english}).
		success(function(data, status) {
			$scope.status = status;
			$scope.data = data;
			$scope.deseret = $sce.trustAsHtml(data.deseret); // Show result from server in our <pre></pre> element
            //alert('got html: ' + $scope.english);
			document.getElementById("input_well").focus()
		})
		.
		error(function(data, status) {
			$scope.data = data || "Request failed";
			$scope.status = status;			
		});
	};

    $scope.deseretToEnglish = function() {

        // Create the http post request
        // the data holds the keywords
        // The request is a JSON request.
        $http.post($scope.url, { "deseret" : $scope.deseret}).
            success(function(data, status) {
                $scope.status = status;
                $scope.data = data;
                $scope.english = $sce.trustAsHtml(data.english); // Show result from server in our <pre></pre> element
                //alert('got html: ' + $scope.english);
                //document.getElementById("output_well").focus()
            })
            .
            error(function(data, status) {
                $scope.data = data || "Request failed";
                $scope.status = status;
            });
    };

}])

app.directive('ngEnter', function () {
    return function (scope, element, attrs) {
        element.bind("keydown keypress", function (event) {
            if(event.which === 13) {
                scope.$apply(function (){
                    scope.$eval(attrs.ngEnter);
                });

                event.preventDefault();
            }
        });
    };
});