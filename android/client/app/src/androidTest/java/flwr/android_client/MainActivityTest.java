package flwr.android_client;

import static androidx.test.espresso.Espresso.onData;
import static androidx.test.espresso.Espresso.onView;
import static androidx.test.espresso.action.ViewActions.click;
import static androidx.test.espresso.action.ViewActions.replaceText;
import static androidx.test.espresso.assertion.ViewAssertions.matches;
import static androidx.test.espresso.matcher.ViewMatchers.isEnabled;
import static androidx.test.espresso.matcher.ViewMatchers.withId;
import static org.hamcrest.Matchers.allOf;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;

import android.util.Log;
import android.view.View;
import androidx.test.espresso.ViewAssertion;
import androidx.test.filters.LargeTest;
import androidx.test.rule.ActivityTestRule;
import androidx.test.runner.AndroidJUnit4;
import org.hamcrest.Matcher;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;

import br.org.eldorado.hiaac.profiling.Profiling;

@RunWith(AndroidJUnit4.class)
@LargeTest
public class MainActivityTest {
    public final String CLIENT = "2";
    public final String IP = "192.168.15.115";
    public final String PORT = "8080";
    public final String EXPERIMENT = "0";
    public final int EXPERIMENT_TIME_MINS = 1;

    @Rule
    public ActivityTestRule<MainActivity> activityRule =
            new ActivityTestRule<>(MainActivity.class);

    @Test
    public void testTrainFederated() {
        Profiling p = Profiling.getInstance();
        p.start();

        onView(withId(R.id.serverIP)).perform(replaceText(IP));
        onView(withId(R.id.serverPort)).perform(replaceText(PORT));
        onView(withId(R.id.device_id_edit_text)).perform(click());

//        waitFor(5000);
        onData(allOf(is(instanceOf(String.class)), is(CLIENT))).perform(click());
        onView(withId(R.id.editTextNumberExpid)).perform(click());
        onData(allOf(is(instanceOf(String.class)), is(EXPERIMENT))).perform(click());
//        waitFor(5000);

        onView(withId(R.id.load_data)).perform(click());
        waitForView(withId(R.id.connect), matches(isEnabled()), 20);
        onView(withId(R.id.connect)).perform(click());
        waitForView(withId(R.id.trainFederated), matches(isEnabled()), 20);
        onView(withId(R.id.trainFederated)).perform(click());
        waitFor(EXPERIMENT_TIME_MINS*60*1000);
        File f = p.finishProfiling();
        Log.d("Test", "File location: " + f.getPath());
    }

    private void waitForView(Matcher<View> viewMatcher, ViewAssertion result, int seconds) {
        for (int i = 1; i <= seconds; i++) {
            try {
                onView(viewMatcher).check(result);
                return;
            } catch (AssertionError e) {
                if (i == seconds) {
                    throw e;
                }
            }
            waitFor(1000);
        }

    }

    private void waitFor(int milis) {
        try {
            Thread.sleep(milis);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}